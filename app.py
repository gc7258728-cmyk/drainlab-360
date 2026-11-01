"""DrainLab 360 — laboratoire pédagogique de pré-dimensionnement pluvial.
Python 3.11+ • aucune donnée personnelle envoyée à un serveur applicatif.
Les données externes ne sont récupérées que sur action explicite de l'utilisateur.
"""
from __future__ import annotations

import html
import io
import json
import math
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, KeepTogether

# ---------- CONSTANTES ET MODELE HYDRAULIQUE ----------
APP_VERSION = "1.0"
LOCAL_TZ = "Africa/Douala"
NAV = ["📊 Vue générale", "🗺️ Mon bassin", "🌧️ Pluviométrie", "📐 Dimensionnement", "🎲 Probabilités", "🎓 Comprendre et tester"]
PALETTE = dict(navy="#11263A", blue="#2789BC", orange="#EAA44B", teal="#218D78", coral="#D8665F", ink="#172C3D", muted="#718496")
COLUMNS = ["date_heure", "pluie_mm"]
EXAMPLE_DAILY = [0, 4, 12, 0, 8, 25, 1, 0, 16, 3]
UNITS = {"pluie_mm": "mm / heure", "intensite_mm_h": "mm/h", "surface_ha": "ha", "debit_m3s": "m³/s"}


def rational_flow(c: float, intensity_mm_h: float, area_ha: float) -> float:
    """Q(m3/s) = C*I(mm/h)*A(ha)/360; coeff constant computed from units."""
    if not (math.isfinite(c) and math.isfinite(intensity_mm_h) and math.isfinite(area_ha) and 0 <= c <= 1 and intensity_mm_h >= 0 and area_ha > 0):
        raise ValueError("C doit être entre 0 et 1 ; I ≥ 0 et A > 0.")
    return c * intensity_mm_h * area_ha / 360.0


def section_geometry(shape: str, width: float, height: float, side: float, reserve_pct: float):
    """Géométrie de la section mouillée à une hauteur tenant compte d'une réserve."""
    if shape not in ("Rectangulaire", "Trapézoïdal"):
        raise ValueError("Section non reconnue.")
    if not all(math.isfinite(float(v)) for v in (width,height,side,reserve_pct)) or width <= 0 or height <= 0 or side < 0 or not 0 <= reserve_pct < 100:
        raise ValueError("Largeur, hauteur, talus et réserve incohérents.")
    y = height * (1 - reserve_pct / 100)
    z = 0.0 if shape == "Rectangulaire" else side
    area = y * (width + z*y)
    perimeter = width + 2*y*math.sqrt(1+z*z)
    top_width = width + 2*z*y
    return area, perimeter, area/perimeter, y, top_width


def manning_capacity(shape: str, width: float, height: float, side: float,
                     slope_pct: float, n: float, reserve_pct: float):
    """Q= (1/n)*A*R^(2/3)*sqrt(S), hypothèse écoulement uniforme, S pente du fond."""
    if not math.isfinite(float(n)) or not math.isfinite(float(slope_pct)) or n <= 0 or slope_pct <= 0 or slope_pct > 100:
        raise ValueError("n et pente doivent être strictement positifs (pente ≤ 100 %).")
    area, perimeter, radius, y, top_width = section_geometry(shape, width, height, side, reserve_pct)
    slope = slope_pct / 100.0
    q = (area/n) * radius**(2/3) * math.sqrt(slope)
    velocity = q/area
    return {"capacite_m3s":q, "surface_m2":area, "perimetre_m":perimeter,
            "rayon_m":radius, "hauteur_m":y, "largeur_surface_m":top_width,
            "vitesse_ms":velocity, "pente_adim":slope}


def validate_project(p: dict) -> None:
    if not str(p.get("nom", "")).strip(): raise ValueError("Nom de bassin obligatoire.")
    if not -90 <= float(p["latitude"]) <= 90 or not -180 <= float(p["longitude"]) <= 180:
        raise ValueError("Coordonnées hors des limites géographiques.")
    if not (math.isfinite(float(p["surface_ha"])) and 0 < float(p["surface_ha"]) <= 1_000_000): raise ValueError("La surface doit être positive.")
    if not (math.isfinite(float(p["coefficient"])) and 0 <= float(p["coefficient"]) <= 1): raise ValueError("Le coefficient C doit être entre 0 et 1.")
    if not (1 <= int(p["duree_h"]) <= 6): raise ValueError("Durée de concentration : 1 à 6 heures dans ce modèle.")


def new_section(name="Caniveau A", shape="Rectangulaire", width=0.30, height=0.30,
                side=0.0, slope_pct=0.5, n=0.015, reserve_pct=0.0):
    return dict(nom=name, forme=shape, largeur_m=width, hauteur_m=height,
                talus=side, pente_pct=slope_pct, rugosite=n, reserve_pct=reserve_pct)


def project_template(name="Nouveau bassin"):
    return dict(id=str(uuid.uuid4()), nom=name, ville="Yaoundé", latitude=3.848, longitude=11.502,
                description="", surface_ha=2.0, coefficient=0.65, duree_h=1,
                intensite_projet=30.0, sections=[new_section()], serie=[],
                source="Aucune donnée chargée", recupere_le="", fictif=False)


def parse_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize 2-column hourly rainfall, preserve holes as NaN, never impute."""
    if not all(c in df.columns for c in COLUMNS):
        raise ValueError("Colonnes obligatoires : date_heure, pluie_mm.")
    if len(df) > 100_000: raise ValueError("Limite de 100 000 lignes pour cette version pédagogique.")
    out = df[COLUMNS].copy()
    out["date_heure"] = pd.to_datetime(out["date_heure"], errors="coerce")
    out["pluie_mm"] = pd.to_numeric(out["pluie_mm"], errors="coerce")
    if out["date_heure"].isna().any(): raise ValueError("Au moins une date/heure est invalide.")
    if out["date_heure"].dt.tz is not None:
        out["date_heure"] = out["date_heure"].dt.tz_convert(LOCAL_TZ).dt.tz_localize(None)
    if (out["pluie_mm"].dropna() < 0).any(): raise ValueError("Les précipitations ne peuvent être négatives.")
    if out["date_heure"].duplicated().any(): raise ValueError("Horodatages dupliqués : corrige le fichier.")
    out = out.sort_values("date_heure").reset_index(drop=True)
    if not out.empty and (out["date_heure"].dt.minute.ne(0) | out["date_heure"].dt.second.ne(0)).any():
        raise ValueError("Les dates doivent correspondre à des heures entières.")
    return out


def with_hour_grid(df: pd.DataFrame):
    """Réindexe aux heures entières, NaN pour les heures absentes, jamais zéro."""
    parsed = parse_weather(df)
    if parsed.empty: return parsed
    idx = pd.date_range(parsed.date_heure.min(), parsed.date_heure.max(), freq="h")
    return parsed.set_index("date_heure").reindex(idx).rename_axis("date_heure").reset_index()


def daily_weather(df: pd.DataFrame, duration_h: int) -> pd.DataFrame:
    """Daily maxima of d-hour moving accumulation/d (mm/h); full d hours mandatory.
    Rolling periods can straddle midnight; date is period end date.
    """
    if duration_h not in range(1,7): raise ValueError("Durée entre 1 et 6 heures.")
    grid = with_hour_grid(df)
    if grid.empty: return pd.DataFrame(columns=["jour", "pluie_totale_mm", "intensite_mm_h", "heures_valides", "complet"])
    grid["intensite_mm_h"] = grid.pluie_mm.rolling(duration_h, min_periods=duration_h).sum() / duration_h
    grid["jour"] = grid.date_heure.dt.floor("D")
    out = grid.groupby("jour", as_index=False).agg(
        pluie_totale_mm=("pluie_mm", lambda x: x.sum(min_count=1)),
        intensite_mm_h=("intensite_mm_h", "max"),
        heures_valides=("pluie_mm", "count"))
    # Full calendar day required to use as independent 24h event sample.
    out["complet"] = out.heures_valides.eq(24)
    out.loc[~out["complet"], ["pluie_totale_mm", "intensite_mm_h"]] = np.nan
    return out


def daily_statistics(daily: pd.DataFrame, rain_threshold: float = 10.0):
    x = pd.to_numeric(daily.get("pluie_totale_mm", pd.Series(dtype=float)), errors="coerce").dropna()
    if x.empty: return dict(n=0, moyenne=np.nan, mediane=np.nan, ecart_type=np.nan,
                            p95=np.nan, depassement=np.nan, nb_depassement=0)
    return dict(n=len(x), moyenne=float(x.mean()), mediane=float(x.median()),
                ecart_type=float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                p95=float(x.quantile(.95)),
                nb_depassement=int((x >= rain_threshold).sum()),
                depassement=float((x >= rain_threshold).mean()))


def valid_blocks(daily: pd.DataFrame, block_days: int):
    """All valid chronological contiguous day blocks, overlap allowed."""
    if block_days <= 0: raise ValueError("La longueur du bloc doit être positive.")
    q = daily.copy().sort_values("jour")
    if q.empty: return np.empty((0,block_days))
    q["jour"] = pd.to_datetime(q.jour)
    values = q.intensite_mm_h.to_numpy(dtype=float)
    dates = q.jour.to_numpy(dtype="datetime64[D]").astype(int)
    output = []
    for i in range(len(values)-block_days+1):
        if np.all(np.isfinite(values[i:i+block_days])) and np.all(np.diff(dates[i:i+block_days]) == 1):
            output.append(values[i:i+block_days])
    return np.array(output, dtype=float).reshape((-1,block_days))


def monte_carlo(daily: pd.DataFrame, project: dict, section: dict,
                simulations: int, horizon: int, block_days: int,
                seed: int, c_uncertainty_pct: float, n_uncertainty_pct: float):
    """Block bootstrap of daily rolling intensities + truncated physical parameter uncertainty.
    Event: at least one simulated day with Q_rational >= Q_Manning.
    No event is retained as count zero, no made-up event date.
    """
    if not 100 <= simulations <= 50_000 or not 1 <= horizon <= 365:
        raise ValueError("Simulations 100–50 000 et horizon 1–365 jours.")
    if simulations * horizon > 2_000_000:
        raise ValueError("Calcul trop volumineux pour un hébergement gratuit : limite 2 millions de journées simulées. Réduis N ou l'horizon.")
    if not 1 <= block_days <= min(horizon, 30): raise ValueError("Longueur de bloc incompatible.")
    blocks = valid_blocks(daily, block_days)
    if not len(blocks): raise ValueError("Pas assez de jours consécutifs complets pour créer des blocs.")
    capacity = manning_capacity(section["forme"], float(section["largeur_m"]),
       float(section["hauteur_m"]),float(section["talus"]),float(section["pente_pct"]),
       float(section["rugosite"]),float(section["reserve_pct"]))["capacite_m3s"]
    rng = np.random.default_rng(seed)
    count_blocks = math.ceil(horizon/block_days)
    selections = rng.integers(0,len(blocks),size=(simulations,count_blocks))
    sampled_rain = blocks[selections].reshape(simulations, count_blocks*block_days)[:,:horizon]
    c = float(project["coefficient"])
    if c_uncertainty_pct > 0:
        c_vals = np.clip(rng.normal(c, c*c_uncertainty_pct/100, simulations),0,1)
    else: c_vals = np.full(simulations,c)
    n = float(section["rugosite"])
    if n_uncertainty_pct > 0:
        n_vals = np.clip(rng.normal(n, n*n_uncertainty_pct/100, simulations), .0001,None)
    else: n_vals = np.full(simulations,n)
    # Manning capacity is inversely proportional to roughness n.
    caps = capacity * n / n_vals
    qs = sampled_rain * (c_vals[:,None]*float(project["surface_ha"])/360)
    counts = (qs >= caps[:,None]).sum(axis=1)
    p = float((counts >= 1).mean())
    # Monte Carlo sampling error only; does not cover model or meteorological uncertainty.
    se = math.sqrt(p*(1-p)/simulations)
    return dict(prob=p, error95=1.96*se, counts=counts, mean=float(counts.mean()),
        quantile90=float(np.quantile(counts,.90,method="higher")),
        no_event=int((counts==0).sum()), n_blocks=len(blocks),
        c_samples=c_vals, n_samples=n_vals)


def evaluation(project, section, intensity=None):
    intensity = project["intensite_projet"] if intensity is None else intensity
    flow = rational_flow(float(project["coefficient"]), float(intensity), float(project["surface_ha"]))
    hyd = manning_capacity(section["forme"],float(section["largeur_m"]),float(section["hauteur_m"]),
                           float(section["talus"]),float(section["pente_pct"]),float(section["rugosite"]),
                           float(section["reserve_pct"]))
    ratio = flow/hyd["capacite_m3s"]
    return dict(debit=flow, taux=ratio, **hyd)


def example_project():
    p = project_template("Bassin-école ENSPY — démonstration")
    p.update(description="Jeu totalement FICTIF : 10 jours, 24 valeurs horaires par jour.",fictif=True,
             source="Exemple fictif intégré (aucune mesure réelle)",
             sections=[new_section("Caniveau compact 20 × 20 cm",width=.20,height=.20),
                       new_section("Caniveau 30 × 30 cm"),
                       new_section("Variante 45 × 40 cm",width=.45,height=.40)])
    start = datetime(2026, 10, 1)
    records=[]
    for idx, total in enumerate(EXAMPLE_DAILY):
        # Split daily total between 12:00 and 14:00 preserving daily total exactly.
        for hour in range(24):
            weight = {12:.2,13:.6,14:.2}.get(hour,0)
            records.append({"date_heure": (start+timedelta(days=idx,hours=hour)).isoformat(timespec="minutes"),
                            "pluie_mm":round(total*weight,7)})
    p["serie"] = records
    return p


def install_weather(project: dict, records: list, source: str, projects: list):
    """Imported/real rain never overwrites an educational demo: create an independent real project."""
    if project.get("fictif"):
        destination=json.loads(json.dumps(project,ensure_ascii=False))
        destination["id"]=str(uuid.uuid4())
        destination["nom"]="Étude météo importée — "+project["ville"]
        destination["fictif"]=False
        projects.append(destination)
    else:
        destination=project
    destination["serie"]=records
    destination["source"]=source
    destination["recupere_le"]=datetime.now(ZoneInfo(LOCAL_TZ)).isoformat(timespec="minutes")
    return destination


def from_api_payload(payload: dict) -> pd.DataFrame:
    hourly=payload.get("hourly")
    if not isinstance(hourly,dict) or "time" not in hourly or "precipitation" not in hourly:
        raise ValueError("La réponse météo ne contient pas les précipitations horaires.")
    if len(hourly["time"]) != len(hourly["precipitation"]):
        raise ValueError("Réponse API incohérente (longueurs différentes).")
    # Open-Meteo precipitation at time T is the total over the PRECEDING hour.
    # Convert to start-of-hour labels used by every internal dataset.
    raw = pd.DataFrame({"date_heure":hourly["time"],"pluie_mm":hourly["precipitation"]})
    raw["date_heure"] = pd.to_datetime(raw["date_heure"],errors="coerce") - pd.Timedelta(hours=1)
    return parse_weather(raw)


@st.cache_data(ttl=60*60*12, show_spinner=False)
def open_meteo_history(lat,lon,start,end,model="era5"):
    # Request one extra day so final desired 23:00–00:00 interval is available.
    extra_end=(date.fromisoformat(end)+timedelta(days=1)).isoformat()
    params={"latitude":lat,"longitude":lon,"start_date":start,"end_date":extra_end,
            "hourly":"precipitation", "timezone":LOCAL_TZ,"precipitation_unit":"mm", "models":model}
    r=requests.get("https://archive-api.open-meteo.com/v1/archive",params=params,timeout=35)
    r.raise_for_status()
    response=r.json()
    if response.get("error"): raise ValueError(response.get("reason", "Erreur Open-Meteo"))
    result=from_api_payload(response)
    first=pd.Timestamp(start)
    last_exclusive=pd.Timestamp(end)+pd.Timedelta(days=1)
    return result[(result.date_heure>=first)&(result.date_heure<last_exclusive)].reset_index(drop=True)


@st.cache_data(ttl=60*60*24, show_spinner=False)
def geocode_city(query):
    r=requests.get("https://geocoding-api.open-meteo.com/v1/search",
         params={"name":query,"count":7,"language":"fr"},timeout=20)
    r.raise_for_status()
    return r.json().get("results",[])


@st.cache_data(ttl=60*60*24, show_spinner=False)
def elevation_srtm(lat,lon):
    r=requests.get("https://api.opentopodata.org/v1/srtm90m",
         params={"locations":f"{lat},{lon}"},timeout=20)
    r.raise_for_status()
    j=r.json()
    if j.get("status") != "OK" or not j.get("results"):
        raise ValueError("Aucune altitude disponible.")
    return j["results"][0].get("elevation")


def to_records(df: pd.DataFrame):
    clean=parse_weather(df)
    return [{"date_heure":row.date_heure.isoformat(timespec="minutes"),
             "pluie_mm":None if pd.isna(row.pluie_mm) else float(row.pluie_mm)}
            for row in clean.itertuples()]


def project_weather(project):
    return parse_weather(pd.DataFrame(project.get("serie",[]),columns=COLUMNS))


def export_json(projects,active_id):
    return json.dumps({"format":"DrainLab360","version":1,"projects":projects,"active_id":active_id},
                      ensure_ascii=False,indent=2).encode("utf-8")


def restore_json(blob):
    if len(blob)>15_000_000: raise ValueError("Sauvegarde trop volumineuse (max. 15 Mo).")
    data=json.loads(blob.decode("utf-8-sig"))
    if data.get("format")!="DrainLab360" or data.get("version")!=1:
        raise ValueError("Format de sauvegarde incompatible.")
    projects=data.get("projects")
    if not isinstance(projects,list) or len(projects)>100: raise ValueError("Liste de projets invalide.")
    ids=set()
    for p in projects:
        validate_project(p)
        if p.get("id") in ids or not p.get("id"): raise ValueError("Identifiants manquants ou dupliqués.")
        ids.add(p["id"])
        if len(p.get("sections",[]))>100: raise ValueError("Trop de variantes.")
        for s in p.get("sections",[]): evaluation(p,s)
        if len(p.get("serie",[]))>100_000: raise ValueError("Trop de lignes météorologiques.")
        project_weather(p)
    active=data.get("active_id")
    return projects, active if active in ids else (projects[0]["id"] if projects else None)


def excel_bytes(project,weather,daily,variants):
    buffer=io.BytesIO()
    with pd.ExcelWriter(buffer,engine="openpyxl") as writer:
        pd.DataFrame([{"Bassin":project["nom"],"Ville":project["ville"],
            "Surface (ha)":project["surface_ha"],"C":project["coefficient"],
            "Durée (h)":project["duree_h"],"Source":project["source"]}]).to_excel(writer,sheet_name="Projet",index=False)
        weather.to_excel(writer,sheet_name="Pluie horaire",index=False)
        daily.to_excel(writer,sheet_name="Analyse journalière",index=False)
        variants.to_excel(writer,sheet_name="Variantes",index=False)
    return buffer.getvalue()


def pdf_bytes(project, daily, variants):
    """Small page-break-safe ReportLab report; font Vera bundled with ReportLab."""
    import reportlab
    root=Path(reportlab.__file__).resolve().parent/"fonts"
    regular=root/"Vera.ttf"; bold=root/"VeraBd.ttf"
    if regular.is_file() and "DrainVera" not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont("DrainVera",str(regular)))
        pdfmetrics.registerFont(TTFont("DrainVeraBold",str(bold)))
    normal_font = "DrainVera" if regular.is_file() else "Helvetica"
    bold_font = "DrainVeraBold" if regular.is_file() else "Helvetica-Bold"
    buffer=io.BytesIO()
    doc=SimpleDocTemplate(buffer,pagesize=(21*cm,29.7*cm),leftMargin=1.8*cm,
       rightMargin=1.8*cm,topMargin=2*cm,bottomMargin=2*cm,
       title="DrainLab 360 - Rapport pédagogique")
    styles=getSampleStyleSheet()
    head=ParagraphStyle("DLHead",parent=styles["Heading1"],fontName=bold_font,
       fontSize=18,leading=23,textColor=colors.HexColor(PALETTE["navy"]),spaceAfter=16)
    sub=ParagraphStyle("DLSub",parent=styles["Heading2"],fontName=bold_font,
       fontSize=11,leading=16,spaceBefore=12,spaceAfter=8,textColor=colors.HexColor(PALETTE["blue"]))
    body=ParagraphStyle("DLBody",fontName=normal_font,fontSize=9.2,leading=15,spaceAfter=8)
    escaped=lambda x: html.escape(str(x))
    story=[Paragraph("DrainLab 360 | Rapport d'étude",head),
      Paragraph("Pré-dimensionnement exploratoire des ouvrages d'assainissement pluvial",body),
      Paragraph("Bassin : "+escaped(project["nom"]),sub),
      Paragraph("Ville : "+escaped(project["ville"])+" | Localisation : "+
        f"{project['latitude']:.4f}, {project['longitude']:.4f}",body),
      Paragraph(f"Surface = {project['surface_ha']} ha ; C = {project['coefficient']} ; "
        f"durée de pluie = {project['duree_h']} h ; intensité de projet = "
        f"{project['intensite_projet']} mm/h.",body),
      Paragraph("Source : "+escaped(project["source"]),body),
      Paragraph("Analyse des pluies",sub)]
    stats=daily_statistics(daily)
    if stats["n"]:
        story.append(Paragraph(f"{stats['n']} jours complets. Pluie journalière moyenne = "
            f"{stats['moyenne']:.2f} mm ; P95 empirique = {stats['p95']:.2f} mm.",body))
    else:story.append(Paragraph("Aucune série horaire complète exploitable.",body))
    story.append(Paragraph("Comparaison de sections",sub))
    if variants.empty: story.append(Paragraph("Aucune variante valide disponible.",body))
    else:
        cells=[["Variante","Q demandé","Q capacité","Q/Qcap","Lecture"]]
        for row in variants.itertuples(index=False):
            cells.append([str(row.Nom)[:27],f"{row.Q_demande_m3s:.4f}",
                 f"{row.Q_capacite_m3s:.4f}",f"{row.Taux:.2f}",str(row.Lecture)[:17]])
        available=17.4*cm
        t=Table(cells,colWidths=[5*cm,2.8*cm,2.8*cm,2.2*cm,4.6*cm],repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("FONTNAME",(0,0),(-1,-1),normal_font),
            ("FONTNAME",(0,0),(-1,0),bold_font),("FONTSIZE",(0,0),(-1,-1),7.4),
            ("BACKGROUND",(0,0),(-1,0),colors.HexColor(PALETTE["navy"])),
            ("TEXTCOLOR",(0,0),(-1,0),colors.white),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#EDF4F8")]),
            ("BOTTOMPADDING",(0,0),(-1,-1),9), ("TOPPADDING",(0,0),(-1,-1),9),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
        story.append(t)
    story.extend([Paragraph("Formules et limites",sub),
        Paragraph("Q = C × I × A / 360 (Q en m³/s, I en mm/h, A en hectares). "
                  "Manning : Q = (1/n) × A_mouillée × R^(2/3) × √S.",body),
        Paragraph("Calcul de pré-étude uniquement : méthode rationnelle limitée aux petits bassins "
                  "et à une intensité associée au temps de concentration. Les intensités extraites "
                  "des données horaires ne constituent PAS une courbe IDF locale validée. "
                  "Écoulement uniforme supposé ; ni pertes singulières, ni remous, ni colmatage, "
                  "ni vérification structurelle, réglementaire ou sécuritaire. "
                  "Faire valider par un professionnel avec données adaptées.",body),
        Paragraph("Rapport généré le "+datetime.now(ZoneInfo(LOCAL_TZ)).strftime("%d/%m/%Y %H:%M")+
                  " — " + ("DONNÉES FICTIVES" if project.get("fictif") else "Données selon source indiquée"),body)])
    def footer(canvas,document):
        canvas.setFont(normal_font,8)
        canvas.setFillColor(colors.HexColor(PALETTE["muted"]))
        canvas.drawString(1.8*cm,1.3*cm,"DrainLab 360 • Projet pédagogique • ENSPY")
        canvas.drawRightString(19.2*cm,1.3*cm,f"Page {document.page}")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return buffer.getvalue()


def dataframe_variants(project,sections):
    rows=[]
    for s in sections:
        try:
            val=evaluation(project,s)
            rows.append({"Nom":s["nom"],"Forme":s["forme"],"Q_demande_m3s":val["debit"],
                "Q_capacite_m3s":val["capacite_m3s"], "Taux":val["taux"],
                "Vitesse_ms":val["vitesse_ms"],"Lecture":"À revoir" if val["taux"]>=1 else "Compatible*"})
        except (ValueError,TypeError,ZeroDivisionError):
            rows.append({"Nom":s.get("nom","?"),"Forme":s.get("forme","?"),
                "Q_demande_m3s":np.nan,"Q_capacite_m3s":np.nan,"Taux":np.nan,
                "Vitesse_ms":np.nan,"Lecture":"Entrée invalide"})
    return pd.DataFrame(rows,columns=["Nom","Forme","Q_demande_m3s","Q_capacite_m3s","Taux","Vitesse_ms","Lecture"])

# ---------- INTERFACE STREAMLIT ----------

def inject_theme(dark=False):
    bg = "#101d2b" if dark else "#F1F5F8"
    card = "#1b3042" if dark else "#FFFFFF"
    txt = "#E7F1F9" if dark else "#172C3D"
    secondary = "#AAC0D0" if dark else "#728496"
    edge = "#365169" if dark else "#DFE8EE"
    st.markdown(f"""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Manrope:wght@500;600;700;800&display=swap');
    html,body,[data-testid="stAppViewContainer"] {{background:{bg};color:{txt};font-family:'DM Sans',system-ui,sans-serif;}}
    [data-testid="stHeader"]{{background:transparent;}}
    .block-container{{padding-top:1.7rem;max-width:1380px;padding-bottom:4rem;}}
    h1,h2,h3,h4{{font-family:'Manrope',sans-serif!important;color:{txt}!important;letter-spacing:-.025em;}}
    p,span,label{{font-family:'DM Sans',sans-serif;}}
    [data-testid="stSidebar"]{{background:#11283D!important;}}
    [data-testid="stSidebar"] *{{color:#D8E5F0;}}
    [data-testid="stSidebar"] [role="radiogroup"] label{{border-radius:11px;padding:7px 7px;}}
    [data-testid="stSidebar"] [role="radiogroup"] label:hover{{background:#21445B;}}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p{{color:#CFDCE7;}}
    [data-testid="stSidebar"] hr{{border-color:#375167;}}
    [data-testid="stMetric"]{{background:{card};border:1px solid {edge};padding:16px;border-radius:14px;}}
    [data-testid="stMetricLabel"]{{color:{secondary};}}
    [data-testid="stMetricValue"]{{color:{txt};font-family:Manrope,sans-serif;}}
    [data-testid="stVerticalBlockBorderWrapper"] > div{{border-radius:15px;}}
    div.stButton>button[kind="primary"],div.stFormSubmitButton>button[kind="primary"] {{background:#EBA445;border:1px solid #EBA445;color:#172B3B;font-weight:800;border-radius:10px;}}
    div.stButton>button,div.stFormSubmitButton>button,div.stDownloadButton>button{{border-radius:10px;font-weight:700;}}
    div.stButton>button:hover,div.stDownloadButton>button:hover{{border-color:#EAA44B;color:#EAA44B;}}
    [data-testid="stDataFrame"], [data-testid="stDataEditor"]{{border-radius:12px;overflow:hidden;}}
    .dl-hero{{border-radius:20px;background:linear-gradient(110deg,#10283F 0%,#1D5679 65%,#298CAF 100%);color:#fff;padding:26px 30px;position:relative;overflow:hidden;box-shadow:0 12px 34px #0c3d5b18;margin-bottom:18px;}}
    .dl-hero:after{{content:'';position:absolute;border-radius:50%;width:230px;height:230px;right:-45px;top:-100px;border:1px solid #ffffff2b;box-shadow:0 0 0 30px #ffffff09,0 0 0 65px #ffffff07;pointer-events:none;}}
    .dl-hero h1{{color:#fff!important;margin:0 0 6px 0;font-size:clamp(1.45rem,3vw,2.25rem);}}
    .dl-hero p{{color:#d7ebf3!important;font-size:14px;max-width:660px;margin:6px 0;}}
    .dl-mini{{font-size:11px;letter-spacing:2px;text-transform:uppercase;font-weight:800;color:#BEE0EC;}}
    .dl-pill{{display:inline-block;border:1px solid #ffffff58;color:#fff;background:#ffffff19;border-radius:999px;padding:5px 10px;font-size:11px;margin-top:9px;}}
    .dl-kpi{{padding:18px 17px;background:{card};border:1px solid {edge};border-radius:15px;box-shadow:0 4px 20px #1a38470b;min-height:129px;margin-bottom:12px;border-top:4px solid var(--accent,#2789BC);}}
    .dl-kpi .label{{font-size:12px;color:{secondary};font-weight:700;}}
    .dl-kpi .value{{font-family:Manrope,sans-serif;font-size:clamp(22px,3vw,31px);font-weight:800;color:{txt};margin-top:9px;letter-spacing:-1px;}}
    .dl-kpi .help{{font-size:11px;color:{secondary};margin-top:3px;}}
    .dl-section{{font-size:12px;letter-spacing:1.4px;font-weight:800;color:#2789BC;margin:12px 0 4px;}}
    .dl-box{{background:{card};padding:17px 20px;border:1px solid {edge};border-radius:14px;margin-bottom:12px;}}
    .dl-box h3{{font-size:16px;margin:0 0 7px 0;}}
    .dl-box p{{color:{secondary};font-size:13px;margin:3px 0;line-height:1.6;}}
    .dl-warning{{padding:12px 15px;border-left:4px solid #EAA44B;background:{'#3d3540' if dark else '#FFF5E6'};border-radius:9px;font-size:13px;color:{txt};}}
    .dl-daystrip{{display:grid;grid-template-columns:repeat(10,minmax(0,1fr));gap:5px;margin:12px 0 3px;}}
    .dl-day{{padding:10px 3px;border-radius:9px;color:white;text-align:center;font-size:11px;font-weight:800;background:#218D78;}}
    .dl-day.alert{{background:#DE9960;}}
    .dl-day.unknown{{background:#93A3B2;}}
    .dl-muted{{color:{secondary};font-size:12px;}}
    @media(max-width:760px){{.block-container{{padding:1rem .75rem 3rem;}}.dl-hero{{padding:20px 17px;border-radius:15px;}}.dl-kpi{{padding:13px 12px;min-height:105px;}}.dl-daystrip{{grid-template-columns:repeat(5,minmax(0,1fr));}}.dl-box{{padding:12px;}}}}
    </style>""",unsafe_allow_html=True)


def card(label, value, subtitle, accent="blue"):
    color=PALETTE.get(accent,accent)
    st.markdown(f'<div class="dl-kpi" style="--accent:{color}">'
                f'<div class="label">{html.escape(str(label))}</div>'
                f'<div class="value">{html.escape(str(value))}</div>'
                f'<div class="help">{html.escape(str(subtitle))}</div></div>',unsafe_allow_html=True)


def hero(title,subtitle,kicker="DRAINLAB 360 · OUTIL DE PRÉ-ÉTUDE",pill=None):
    p=f'<span class="dl-pill">{html.escape(pill)}</span>' if pill else ''
    st.markdown(f'<div class="dl-hero"><div class="dl-mini">{html.escape(kicker)}</div>'
                f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>{p}</div>',unsafe_allow_html=True)


def note(text):
    st.markdown(f'<div class="dl-warning">{html.escape(text)}</div>',unsafe_allow_html=True)


def active_project():
    return next((p for p in st.session_state.projects if p["id"]==st.session_state.active_id),None)


def choose_project(identifier):
    st.session_state.active_id=identifier


def goto(page):
    st.session_state.page=page


def add_demo():
    existing=next((p for p in st.session_state.projects if p.get("fictif") and
                   p.get("nom")=="Bassin-école ENSPY — démonstration"),None)
    if existing: st.session_state.active_id=existing["id"]
    else:
        project=example_project()
        st.session_state.projects.append(project)
        st.session_state.active_id=project["id"]
    st.session_state.page=NAV[0]


def reset_demo():
    removed=[p["id"] for p in st.session_state.projects if p.get("fictif")]
    st.session_state.projects=[p for p in st.session_state.projects if not p.get("fictif")]
    if st.session_state.active_id in removed:
        st.session_state.active_id=st.session_state.projects[0]["id"] if st.session_state.projects else None
    st.session_state.page=NAV[0]


def make_plot(fig,height=325):
    fig.update_layout(template="plotly_white",height=height,margin=dict(l=12,r=12,t=35,b=14),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial",color=PALETTE["ink"]),legend=dict(orientation="h",y=1.12,x=0))
    st.plotly_chart(fig,use_container_width=True,config={"displaylogo":False,"responsive":True})


def section_lookup(project):
    return {s["nom"]:s for s in project["sections"]}


def daily_for(project):
    return daily_weather(project_weather(project),int(project["duree_h"]))


def safe_daily(project):
    try:return daily_for(project)
    except Exception as exc:
        st.error(f"Données météorologiques impossibles à interpréter : {exc}")
        return pd.DataFrame(columns=["jour","pluie_totale_mm","intensite_mm_h","heures_valides","complet"])


def show_project_picker():
    p=active_project()
    if p is None:
        st.info("Aucun bassin ouvert. Crée un bassin ou charge l'exemple fictif.")
        c1,c2=st.columns(2)
        with c1: st.button("➕ Créer un bassin",on_click=goto,args=(NAV[1],),use_container_width=True)
        with c2: st.button("🧪 Charger l'exemple fictif",on_click=add_demo,use_container_width=True)
    return p


def dashboard():
    p=active_project()
    hero("Des données aux décisions hydrauliques.",
         "Évaluez le ruissellement, comparez les sections et quantifiez les dépassements de capacité.",
         pill="ENSPY · Génie civil · Probabilités & hydraulique")
    if not p:
        st.markdown('<div class="dl-box"><h3>Bienvenue dans votre laboratoire hydraulique</h3>'
            '<p>Commencez avec un bassin fictif entièrement vérifiable, ou créez votre propre étude.</p></div>',
            unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1: st.button("🧪 Charger l'exemple fictif",on_click=add_demo,type="primary",use_container_width=True)
        with c2: st.button("➕ Nouveau bassin",on_click=goto,args=(NAV[1],),use_container_width=True)
        return
    daily=safe_daily(p)
    stats=daily_statistics(daily)
    variants=dataframe_variants(p,p["sections"])
    valid=variants.dropna(subset=["Taux"])
    demand=rational_flow(p["coefficient"],p["intensite_projet"],p["surface_ha"])
    if p.get("fictif"): st.warning("🧪 DONNÉES FICTIVES — démonstration pédagogique, sans observations réelles.")
    st.markdown(f'<div class="dl-section">BASSIN ACTIF · {html.escape(p["nom"].upper())}</div>',unsafe_allow_html=True)
    cols=st.columns(4)
    with cols[0]:card("Surface du bassin",f"{p['surface_ha']:,.2f} ha","Surface contributive","blue")
    with cols[1]:card("Débit estimé",f"{demand:.3f} m³/s",f"I = {p['intensite_projet']:g} mm/h","orange")
    with cols[2]:card("Jours analysables",str(stats["n"]),"Jours horaires complets","teal")
    with cols[3]:card("Variantes évaluées",str(len(valid)),f"{len(valid[valid.Taux>=1])} capacités dépassées","coral")
    first,second=st.columns([1.4,1],gap="medium")
    with first:
        st.subheader("🌧️ Pluie et intensités")
        valid_days=daily.dropna(subset=["pluie_totale_mm"])
        if valid_days.empty:
            st.info("Charge des pluies horaires dans « Pluviométrie » pour afficher les courbes.")
        else:
            fig=px.bar(valid_days,x="jour",y="pluie_totale_mm",labels={"jour":"Jour","pluie_totale_mm":"Pluie (mm/j)"})
            fig.update_traces(marker_color=PALETTE["blue"])
            make_plot(fig,280)
    with second:
        st.subheader("📐 Comparaison hydraulique")
        if valid.empty:st.info("Ajoute des sections valides dans « Dimensionnement ».")
        else:
            fig=px.bar(valid,x="Nom",y="Taux",color="Lecture",labels={"Nom":"Section","Taux":"Q demandé / Q capacité"},
                  color_discrete_map={"À revoir":PALETTE["coral"],"Compatible*":PALETTE["teal"]})
            fig.add_hline(y=1,line_dash="dash",line_color=PALETTE["orange"],annotation_text="Capacité = demande")
            make_plot(fig,280)
    st.subheader("🗓️ Calendrier des pluies")
    if stats["n"]:
        values=daily.head(30)
        tiles=[]
        for row in values.itertuples():
            css="unknown" if pd.isna(row.pluie_totale_mm) else ("alert" if row.pluie_totale_mm>=10 else "")
            tooltip="Données absentes" if pd.isna(row.pluie_totale_mm) else f"{row.pluie_totale_mm:.1f} mm"
            tiles.append(f'<div class="dl-day {css}" title="{html.escape(tooltip)}">{pd.Timestamp(row.jour):%d/%m}</div>')
        st.markdown('<div class="dl-daystrip">'+''.join(tiles)+'</div>',unsafe_allow_html=True)
        st.caption("Vert : pluie < 10 mm/j · Orange : pluie ≥ 10 mm/j · Gris : jour incomplet. Seuil pédagogique modifiable dans l'analyse.")
    st.markdown("### Actions rapides")
    a,b,c,d=st.columns(4)
    with a:st.button("🗺️ Mon bassin",on_click=goto,args=(NAV[1],),use_container_width=True)
    with b:st.button("🌧️ Charger des pluies",on_click=goto,args=(NAV[2],),use_container_width=True)
    with c:st.button("📐 Dimensionner",on_click=goto,args=(NAV[3],),use_container_width=True)
    with d:st.button("🎲 Simuler",on_click=goto,args=(NAV[4],),use_container_width=True)
    with st.expander("📥 Rapport et sauvegardes"):
        pdf=pdf_bytes(p,daily,variants)
        st.download_button("Télécharger le rapport PDF",pdf,file_name="DrainLab360_rapport.pdf",mime="application/pdf")
        st.download_button("Sauvegarder tous les bassins (JSON)",
             export_json(st.session_state.projects,st.session_state.active_id),
             file_name="DrainLab360_sauvegarde.json",mime="application/json")
    note("Les valeurs de pré-dimensionnement ne remplacent pas les données IDF locales, un levé topographique ni une vérification réglementaire.")


def basin_page():
    hero("Mes bassins versants","Crée, modifie et retrouve tes études ; la surface et le coefficient de ruissellement restent configurables.","ÉTUDES · LOCALISATION · HYPOTHÈSES")
    st.subheader("🗂️ Études enregistrées")
    projects=st.session_state.projects
    if not projects:st.info("Aucun bassin enregistré dans la session.")
    else:
        for group_start in range(0,len(projects),3):
            cols=st.columns(3)
            for i,p in enumerate(projects[group_start:group_start+3]):
                with cols[i]:
                    with st.container(border=True):
                        st.markdown(f"**{html.escape(p['nom'])}**")
                        st.caption(f"📍 {p['ville']} · {p['surface_ha']:g} ha · C = {p['coefficient']:g}")
                        if p.get("fictif"): st.caption("🧪 Exemple fictif")
                        st.button("Ouvrir cette étude",key=f"open_{p['id']}",
                            on_click=choose_project,args=(p["id"],),use_container_width=True)
    with st.expander("➕ Créer un nouveau bassin",expanded=not bool(projects)):
        with st.form("create_basin",clear_on_submit=True):
            name=st.text_input("Nom de l'étude *",placeholder="Ex. Drainage quartier A")
            c1,c2=st.columns(2)
            with c1:city=st.text_input("Ville",value="Yaoundé")
            with c2:area=st.number_input("Surface contributive (ha)",min_value=.001,max_value=1_000_000.,value=2.0,step=.1)
            c1,c2,c3=st.columns(3)
            with c1:lat=st.number_input("Latitude (°)",min_value=-90.0,max_value=90.0,value=3.848,format="%.5f")
            with c2:lon=st.number_input("Longitude (°)",min_value=-180.0,max_value=180.0,value=11.502,format="%.5f")
            with c3:c=st.number_input("Coefficient C (sans unité)",min_value=0.,max_value=1.,value=.65,step=.05)
            desc=st.text_area("Description",max_chars=500)
            submitted=st.form_submit_button("Créer le bassin",type="primary")
            if submitted:
                if not name.strip():st.error("Indique un nom.")
                else:
                    p=project_template(name.strip())
                    p.update(ville=city.strip(),surface_ha=area,latitude=lat,longitude=lon,coefficient=c,description=desc)
                    st.session_state.projects.append(p);st.session_state.active_id=p["id"]
                    st.success("Bassin créé.");st.rerun()
    p=show_project_picker()
    if p is None:return
    st.markdown("### ✏️ Paramètres du bassin sélectionné")
    if p.get("fictif"):st.info("Tu modifies une copie pédagogique en session ; tes autres bassins restent séparés.")
    with st.form("edit_basin"):
        name=st.text_input("Nom",p["nom"])
        c1,c2=st.columns(2)
        with c1: city=st.text_input("Ville / secteur",p["ville"])
        with c2: surface=st.number_input("Surface A (ha)",value=float(p["surface_ha"]),min_value=.001,step=.1)
        c1,c2,c3=st.columns(3)
        with c1: coeff=st.number_input("Coefficient C",value=float(p["coefficient"]),min_value=0.,max_value=1.,step=.05)
        with c2: duration=st.selectbox("Durée d'étude d (h)",[1,2,3,4,5,6],index=int(p["duree_h"])-1)
        with c3: design_int=st.number_input("Intensité de projet I (mm/h)",min_value=0.,max_value=1000.,value=float(p["intensite_projet"]),step=1.)
        c1,c2=st.columns(2)
        with c1:lat=st.number_input("Latitude (°)",min_value=-90.,max_value=90.,value=float(p["latitude"]),format="%.5f")
        with c2:lon=st.number_input("Longitude (°)",min_value=-180.,max_value=180.,value=float(p["longitude"]),format="%.5f")
        descr=st.text_area("Description du bassin",value=p["description"],max_chars=500)
        save=st.form_submit_button("Enregistrer les modifications",type="primary")
        if save:
            candidate={**p,"nom":name.strip(),"ville":city.strip(),"latitude":lat,
                "longitude":lon,"surface_ha":surface,"coefficient":coeff,"duree_h":duration,
                "intensite_projet":design_int,"description":descr}
            try:validate_project(candidate)
            except ValueError as e: st.error(str(e))
            else:p.update(candidate);st.success("Modifications enregistrées. Les analyses utilisent les nouveaux paramètres.");st.rerun()
    c1,c2=st.columns(2)
    with c1:
        search=st.text_input("Chercher une ville (Open-Meteo)",value=p["ville"],key="city_search")
        if st.button("🔎 Rechercher la localisation"):
            try:
                matches=geocode_city(search)
                if not matches:st.warning("Aucune localité correspondante.")
                else:st.session_state.city_results=matches
            except (ValueError,requests.RequestException) as e:st.error(f"Localisation inaccessible : {e}")
        if st.session_state.get("city_results"):
            matches=st.session_state.city_results
            labels=[f"{x['name']} — {x.get('admin1','')}, {x.get('country','')}" for x in matches]
            selected=st.selectbox("Choisir un résultat",range(len(matches)),format_func=lambda i:labels[i])
            if st.button("Appliquer ces coordonnées"):
                x=matches[selected];p.update(ville=x["name"],latitude=float(x["latitude"]),longitude=float(x["longitude"]))
                st.session_state.city_results=[];st.success("Coordonnées enregistrées.");st.rerun()
    with c2:
        st.caption("Altitude indicative (MNT SRTM, ne fournit PAS la pente d'un ouvrage)")
        if st.button("⛰️ Interroger l'altitude"):
            try:
                alt=elevation_srtm(p["latitude"],p["longitude"])
                st.info("Altitude estimée : "+(f"{alt} m" if alt is not None else "non disponible"))
            except (ValueError,requests.RequestException) as e:st.error(f"Service topo indisponible : {e}")
    st.map(pd.DataFrame({"lat":[p["latitude"]],"lon":[p["longitude"]]}),zoom=10)
    with st.expander("🗑️ Suppression du bassin"):
        confirmation=st.checkbox("Je confirme vouloir supprimer ce bassin et ses données de session.")
        if st.button("Supprimer définitivement cette étude",disabled=not confirmation):
            st.session_state.projects=[x for x in projects if x["id"]!=p["id"]]
            st.session_state.active_id=st.session_state.projects[0]["id"] if st.session_state.projects else None
            st.success("Bassin supprimé.");st.rerun()


def rainfall_page():
    hero("Pluviométrie historique","Étudiez les précipitations horaires, les intensités glissantes et la qualité de l'échantillon.","DONNÉES · STATISTIQUES · SOURCES")
    p=show_project_picker()
    if p is None:return
    if p.get("fictif"):st.warning("Données fictives de démonstration : elles ne décrivent pas les pluies réelles de Yaoundé.")
    with st.expander("🌐 Récupérer les pluies horaires — Open-Meteo",expanded=not bool(p["serie"])):
        st.caption("Source : réanalyse ERA5, non une station pluviométrique. Le serveur renvoie des heures locales Africa/Douala. Les jours récents peuvent être indisponibles.")
        yesterday=date.today()-timedelta(days=10)
        start_default=yesterday-timedelta(days=365)
        with st.form("api_form"):
            a,b=st.columns(2)
            with a:start=st.date_input("Début de période",value=start_default,min_value=date(1940,1,1))
            with b:end=st.date_input("Fin de période",value=yesterday,min_value=date(1940,1,1))
            submit=st.form_submit_button("Télécharger depuis Open-Meteo",type="primary")
            if submit:
                if end<start:st.error("La date de fin doit suivre le début.")
                elif (end-start).days>3650:st.error("Sélectionne 10 ans maximum par requête.")
                elif end>date.today()-timedelta(days=6):st.error("Pour ERA5 et l'heure additionnelle, sélectionne une fin datant d'au moins six jours.")
                else:
                    try:
                        with st.spinner("Récupération des données horaires..."):
                            fetched=open_meteo_history(p["latitude"],p["longitude"],start.isoformat(),end.isoformat())
                        if fetched.empty:st.warning("Aucune mesure retournée.")
                        else:
                            dest=install_weather(p,to_records(fetched),
                                f"Open-Meteo ERA5 (réanalyse), {start} au {end}",st.session_state.projects)
                            st.session_state.active_id=dest["id"]
                            st.success(f"{len(fetched):,} heures reçues. Données actualisées.");st.rerun()
                    except (requests.RequestException,ValueError,KeyError) as exc:
                        st.error(f"Le chargement a échoué : {exc}. Tu peux importer un CSV/Excel.")
    with st.expander("📁 Importer un CSV ou Excel"):
        st.caption("Colonnes : date_heure (AAAA-MM-JJ HH:00, DÉBUT de l’heure locale) et pluie_mm (cumul de cette heure, en mm). Une ligne par heure. Les manquants restent manquants.")
        uploaded=st.file_uploader("Fichier de pluie",type=["csv","xlsx"],key="rain_upload")
        if uploaded and st.button("Importer et valider le fichier"):
            try:
                if uploaded.name.lower().endswith(".csv"):
                    data=pd.read_csv(uploaded,sep=None,engine="python")
                else:data=pd.read_excel(uploaded)
                parsed=parse_weather(data)
                if parsed.empty:raise ValueError("Fichier vide.")
                dest=install_weather(p,to_records(parsed),f"Import utilisateur : {uploaded.name}",st.session_state.projects)
                st.session_state.active_id=dest["id"]
                st.success(f"{len(parsed)} heures importées. Les analyses sont recalculées.");st.rerun()
            except Exception as exc:st.error(f"Import impossible : {exc}")
    weather=project_weather(p)
    st.markdown("### Données disponibles")
    st.caption(f"**Source :** {p['source']} · **Chargées le :** {p.get('recupere_le') or 'non renseigné'} · Fuseau : {LOCAL_TZ} · Horodatage interne = début de l’heure")
    if weather.empty:
        st.info("Aucune donnée. Lance le téléchargement ou importe un fichier pour commencer l'analyse.")
        return
    # Duration can be changed globally on the basin page; dynamic below makes it easy to test.
    duration=st.select_slider("Durée de cumul glissant d (h) pour l'analyse (non enregistrée)",
       options=[1,2,3,4,5,6],value=int(p["duree_h"]))
    threshold=st.number_input("Seuil de pluie journalière pour la fréquence (mm/j)",
       min_value=0.,max_value=1000.,value=10.,step=1.)
    daily=daily_weather(weather,duration)
    stat=daily_statistics(daily,threshold)
    good=daily.dropna(subset=["pluie_totale_mm"])
    a,b,c,d=st.columns(4)
    with a:card("Jours complets",str(stat["n"]),"24 heures valides", "blue")
    with b:card("Pluie moyenne",f"{stat['moyenne']:.2f} mm" if stat["n"] else "—","Par jour valide", "teal")
    with c:card("Percentile 95",f"{stat['p95']:.2f} mm" if stat["n"] else "—","Empirique et descriptif", "orange")
    with d:card("Fréquence ≥ seuil",f"{stat['depassement']:.1%}" if stat["n"] else "—",f"{stat['nb_depassement']} jours", "coral")
    if stat["n"]:
        x,y=st.columns(2)
        with x:
            fig=px.bar(good,x="jour",y="pluie_totale_mm",labels={"jour":"Date","pluie_totale_mm":"Pluie (mm/j)"},title="Pluie journalière")
            fig.update_traces(marker_color=PALETTE["blue"])
            fig.add_hline(y=threshold,line_color=PALETTE["orange"],line_dash="dash")
            make_plot(fig)
        with y:
            fig=px.histogram(good,x="intensite_mm_h",nbins=min(25,max(4,len(good)//2)),
                title=f"Pics d'intensité sur {duration} h",labels={"intensite_mm_h":"Intensité (mm/h)"})
            fig.update_traces(marker_color=PALETTE["teal"])
            make_plot(fig)
        monthly=good.assign(mois=good.jour.dt.strftime("%Y-%m")).groupby("mois",as_index=False)["pluie_totale_mm"].sum()
        fig=px.bar(monthly,x="mois",y="pluie_totale_mm",title="Cumuls mensuels des jours complets",
             labels={"mois":"Mois","pluie_totale_mm":"Total (mm)"})
        fig.update_traces(marker_color=PALETTE["orange"])
        make_plot(fig,290)
    incomplete=int((~daily.complet).sum())
    if incomplete:st.warning(f"{incomplete} jours incomplets exclus des statistiques (moins de 24 heures valides).")
    with st.expander("🔍 Voir les données brutes et les calculs",expanded=True):
        tab1,tab2=st.tabs(["Données brutes", "Bilan journalier"])
        with tab1:st.dataframe(weather,use_container_width=True,height=250)
        with tab2:st.dataframe(daily,use_container_width=True,height=250)
        st.download_button("⬇️ Exporter les pluies horaires (CSV)",weather.to_csv(index=False).encode("utf-8-sig"),
             file_name="DrainLab360_pluie_horaire.csv",mime="text/csv")
        st.download_button("⬇️ Exporter le bilan journalier (CSV)",daily.to_csv(index=False).encode("utf-8-sig"),
             file_name="DrainLab360_bilan_journalier.csv",mime="text/csv")
    st.caption("Important : les intensités calculées à partir de cumuls horaires servent à une analyse exploratoire. Elles ne constituent pas une courbe IDF de projet.")


def design_page():
    hero("Pré-dimensionnement hydraulique","Faites varier la géométrie, la pente et la rugosité ; comparez le débit demandé à la capacité théorique.","CALCULS · MANNING · VARIANTES")
    p=show_project_picker()
    if p is None:return
    note("Pré-étude, pas validation de chantier : l'intensité de projet doit être justifiée par une durée de concentration et des données IDF locales.")
    st.markdown("### Hypothèses hydrologiques")
    with st.form("demand_form"):
        c1,c2,c3,c4=st.columns(4)
        with c1:area=st.number_input("Surface A (ha)",min_value=.001,value=float(p["surface_ha"]),step=.1)
        with c2:coef=st.number_input("Coefficient C",min_value=0.,max_value=1.,value=float(p["coefficient"]),step=.05)
        with c3:intensity=st.number_input("Intensité I (mm/h)",min_value=0.,max_value=1000.,value=float(p["intensite_projet"]),step=1.)
        with c4:duration=st.selectbox("Durée = tc (h), hypothèse",[1,2,3,4,5,6],index=int(p["duree_h"])-1)
        apply=st.form_submit_button("Recalculer et enregistrer les hypothèses",type="primary")
        if apply:
            p.update(surface_ha=area,coefficient=coef,intensite_projet=intensity,duree_h=duration)
            st.success("Hypothèses enregistrées ; les résultats sont actualisés.");st.rerun()
    q=rational_flow(p["coefficient"],p["intensite_projet"],p["surface_ha"])
    a,b,c=st.columns(3)
    with a:card("Débit de ruissellement",f"{q:.4f} m³/s","Méthode rationnelle", "blue")
    with b:card("Intensité de projet",f"{p['intensite_projet']:g} mm/h",f"Durée supposée = {p['duree_h']} h", "orange")
    with c:card("Surface contributive",f"{p['surface_ha']:g} ha",f"C = {p['coefficient']:g}", "teal")
    st.latex(r"Q_{r}=\frac{C\cdot I\cdot A_{ha}}{360}\quad(\mathrm{m^3/s})")
    st.subheader("Sections et scénarios")
    st.caption("Édite le tableau puis clique sur « Enregistrer les sections ». Une réserve de hauteur réduit la lame d'eau hydraulique disponible. Pente en % (0,5 = 0,005 m/m). Talus z = horizontal/vertical.")
    sections=pd.DataFrame(p["sections"])
    edited=st.data_editor(sections,num_rows="dynamic",use_container_width=True,hide_index=True,
        column_config={
          "nom":st.column_config.TextColumn("Nom de variante",required=True),
          "forme":st.column_config.SelectboxColumn("Forme",options=["Rectangulaire","Trapézoïdal"],required=True),
          "largeur_m":st.column_config.NumberColumn("Fond b (m)",min_value=.01,step=.05,format="%.2f"),
          "hauteur_m":st.column_config.NumberColumn("Hauteur H (m)",min_value=.01,step=.05,format="%.2f"),
          "talus":st.column_config.NumberColumn("Talus z (H/V)",min_value=0.,step=.25,format="%.2f"),
          "pente_pct":st.column_config.NumberColumn("Pente S (%)",min_value=.001,max_value=100.,step=.1,format="%.3f"),
          "rugosite":st.column_config.NumberColumn("Manning n",min_value=.001,step=.001,format="%.4f"),
          "reserve_pct":st.column_config.NumberColumn("Réserve (%)",min_value=0.,max_value=99.,step=5.)
        },key=f"sections_editor_{p['id']}")
    edited_sections=edited.to_dict(orient="records")
    issues=[]
    for i,s in enumerate(edited_sections,1):
        try:
            if not str(s.get("nom","")).strip():raise ValueError("nom obligatoire")
            evaluation(p,s)
        except (ValueError,TypeError,KeyError,ZeroDivisionError) as exc:issues.append(f"Ligne {i} : {exc}")
    valid_names=[str(s.get("nom","")).strip() for s in edited_sections]
    if len(valid_names)!=len(set(valid_names)):
        issues.append("Chaque section doit avoir un nom différent.")
    if issues:
        for msg in issues:st.error(msg)
    c1,c2=st.columns([1,1])
    with c1:
        if st.button("💾 Enregistrer les sections",type="primary",disabled=bool(issues)):
            if not edited_sections:st.warning("Au moins une section est nécessaire pour la comparaison.")
            else:
                p["sections"]=edited_sections
                st.success("Scénarios enregistrés.");st.rerun()
    with c2:
        if st.button("➕ Ajouter une variante prête à modifier"):
            count=len(p["sections"])+1
            p["sections"].append(new_section(f"Variante {count}",width=.4,height=.4))
            st.rerun()
    variants=dataframe_variants(p,edited_sections)
    st.markdown("### Résultats recalculés à partir du tableau visible")
    st.dataframe(variants.style.format({"Q_demande_m3s":"{:.4f}","Q_capacite_m3s":"{:.4f}",
        "Taux":"{:.2f}","Vitesse_ms":"{:.2f}"},na_rep="—"),use_container_width=True,hide_index=True)
    valid=variants.dropna(subset=["Taux"])
    if not valid.empty:
        fig=go.Figure()
        fig.add_bar(name="Débit demandé",x=valid["Nom"],y=valid["Q_demande_m3s"],marker_color=PALETTE["orange"])
        fig.add_bar(name="Capacité hydraulique",x=valid["Nom"],y=valid["Q_capacite_m3s"],marker_color=PALETTE["teal"])
        fig.update_layout(barmode="group",yaxis_title="Débit (m³/s)")
        make_plot(fig)
        target=st.selectbox("Examiner une section",range(len(valid)),format_func=lambda i: str(valid.iloc[i]["Nom"]))
        row=valid.iloc[target]
        details=evaluation(p,edited_sections[variants.index[variants["Nom"]==row.Nom][0]])
        status="DÉPASSEMENT" if details["taux"]>=1 else "COMPATIBLE SUR CE CRITÈRE"
        st.info(f"**{row.Nom}** · {status}. Q demandé / Q capacité = {details['taux']:.2f}. "
                f"Rayon hydraulique = {details['rayon_m']:.3f} m ; vitesse = {details['vitesse_ms']:.2f} m/s. "
                "La vitesse doit encore être vérifiée selon le matériau, l'érosion et les règles applicables.")
    st.latex(r"Q_{Manning}=\frac{1}{n}\,A_{mouillée}\,R^{2/3}\sqrt{S}\quad;\quad R=\frac{A_{mouillée}}{P_{mouillé}}")
    with st.expander("📦 Exports complets de l'étude"):
        weather=project_weather(p);daily=safe_daily(p)
        st.download_button("📄 Rapport PDF",pdf_bytes(p,daily,variants),
            file_name="DrainLab360_rapport.pdf",mime="application/pdf")
        st.download_button("📊 Données et résultats Excel",excel_bytes(p,weather,daily,variants),
            file_name="DrainLab360_analyse.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        st.download_button("📁 Tableau des scénarios CSV",variants.to_csv(index=False).encode("utf-8-sig"),
            file_name="DrainLab360_scenarios.csv",mime="text/csv")
    st.caption("* Compatible signifie uniquement que Q demandé < Q capacité dans les hypothèses définies. Ce n'est pas un certificat de sécurité.")


def probability_page():
    hero("Probabilités de dépassement","Observez les fréquences historiques et simulez l'incertitude sur la pluie, C et la rugosité de Manning.","MONTE-CARLO · BLOCS · SENSIBILITÉ")
    p=show_project_picker()
    if p is None:return
    if not p["serie"]:
        st.info("Ajoute d'abord une série horaire dans « Pluviométrie », ou charge l'exemple fictif.")
        return
    daily=safe_daily(p)
    if not p["sections"]:
        st.info("Ajoute une section dans « Dimensionnement ».");return
    sections=section_lookup(p)
    sec_name=st.selectbox("Section analysée",list(sections))
    s=sections[sec_name]
    try:cap=evaluation(p,s)["capacite_m3s"]
    except ValueError as exc:st.error(str(exc));return
    complete=daily.dropna(subset=["intensite_mm_h"])
    if complete.empty:st.warning("Aucune journée complète ; une série horaire est nécessaire.");return
    threshold_i=cap*360/(float(p["coefficient"])*float(p["surface_ha"])) if p["coefficient"]>0 else math.inf
    demand=complete.intensite_mm_h.to_numpy(float)*p["coefficient"]*p["surface_ha"]/360
    event=demand>=cap
    p_obs=float(event.mean())
    a,b,c=st.columns(3)
    with a:card("Fréquence historique",f"{p_obs:.1%}",f"{int(event.sum())} / {len(event)} jours", "coral")
    with b:card("Capacité de la section",f"{cap:.4f} m³/s","Selon Manning", "teal")
    with c:card("Intensité limite équiv.",f"{threshold_i:.1f} mm/h" if np.isfinite(threshold_i) else "∞","Q rational = Q Manning", "orange")
    st.caption(f"Événement exact : sur une journée complète, le maximum des cumuls glissants de {p['duree_h']} heure(s) "
        "produit un débit rationnel supérieur ou égal à la capacité. Le résultat est une fréquence historique exploratoire, pas une probabilité annuelle d'inondation.")
    fig=px.histogram(complete,x="intensite_mm_h",nbins=min(max(5,len(complete)//2),40),
        title="Intensités maximales journalières",labels={"intensite_mm_h":"mm/h"})
    fig.update_traces(marker_color=PALETTE["blue"])
    if math.isfinite(threshold_i):fig.add_vline(x=threshold_i,line_dash="dash",line_color=PALETTE["coral"])
    make_plot(fig)
    st.subheader("🎲 Simulation de Monte-Carlo par blocs")
    st.caption("Échantillonnage avec remise de blocs de jours consécutifs complets. La corrélation interne aux blocs est partiellement conservée. Les simulations sans dépassement sont conservées.")
    with st.form("monte_form"):
        x,y,z=st.columns(3)
        with x:n_sims=st.number_input("Nombre de simulations",min_value=100,max_value=50000,value=3000,step=500)
        with y:horizon=st.number_input("Horizon simulé (jours)",min_value=1,max_value=365,value=10)
        with z:block=st.number_input("Taille de bloc (jours)",min_value=1,max_value=30,value=3)
        a,b,c=st.columns(3)
        with a:seed=st.number_input("Graine aléatoire",min_value=0,max_value=2**31-1,value=42,step=1)
        with b:unc_c=st.slider("Incertitude relative sur C (%)",0,50,10)
        with c:unc_n=st.slider("Incertitude relative sur n (%)",0,50,10)
        run=st.form_submit_button("Lancer la simulation",type="primary")
    # Only retain output with matching params and exact data source signature.
    import hashlib
    signature=hashlib.sha256(json.dumps({"id":p["id"],"serie":p["serie"],"section":s,
            "surface":p["surface_ha"],"c":p["coefficient"],"d":p["duree_h"],
            "settings":[n_sims,horizon,block,seed,unc_c,unc_n]},sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    if run:
        try:
            result=monte_carlo(daily,p,s,int(n_sims),int(horizon),int(block),int(seed),float(unc_c),float(unc_n))
            st.session_state.simulation=(signature,result)
        except ValueError as exc:st.error(f"Simulation impossible : {exc}")
    current=st.session_state.get("simulation")
    if current and current[0]==signature:
        r=current[1]
        cols=st.columns(4)
        with cols[0]:card("P simulée ≥ 1 dépassement",f"{r['prob']:.1%}","Sur l'horizon choisi", "coral")
        with cols[1]:card("Marge ±95% MC",f"±{r['error95']:.1%}","Erreur d'échantillonnage seule", "orange")
        with cols[2]:card("Jours dépassés (moy.)",f"{r['mean']:.2f}","Dans chaque trajectoire", "blue")
        with cols[3]:card("Trajectoires sans événement",str(r["no_event"]),f"Sur {int(n_sims)} simulations", "teal")
        hist=pd.DataFrame({"Nombre de jours avec dépassement":r["counts"]})
        fig=px.histogram(hist,x="Nombre de jours avec dépassement",title="Distribution du nombre de jours en dépassement")
        fig.update_traces(marker_color=PALETTE["orange"])
        make_plot(fig)
        st.info(f"Quantile 90 % (méthode « higher ») : {int(r['quantile90'])} jours. "
                f"{r['n_blocks']} blocs historiques admissibles. Graine {int(seed)} : résultat reproductible. "
                "L'intervalle ±95 % est une approximation de l'erreur due au nombre de simulations uniquement.")
    else:
        st.info("Règle les paramètres puis clique sur « Lancer la simulation ». Après tout changement, un nouveau calcul est nécessaire.")
    st.subheader("🎚️ Sensibilité à l'intensité de pluie")
    if not math.isfinite(threshold_i):st.warning("Coefficient C nul : le débit rationnel sera nul quelle que soit l'intensité.")
    else:
        values=np.linspace(max(0,threshold_i*.2),max(10,threshold_i*1.8),35)
        sens=pd.DataFrame({"Intensité (mm/h)":values,
            "Q calculé (m³/s)":p["coefficient"]*p["surface_ha"]*values/360})
        fig=px.line(sens,x="Intensité (mm/h)",y="Q calculé (m³/s)",markers=True)
        fig.add_hline(y=cap,line_dash="dash",line_color=PALETTE["coral"],annotation_text="Capacité")
        make_plot(fig,290)
    with st.expander("📚 Hypothèses à citer lors de la soutenance"):
        st.markdown("""- **Fréquence empirique :** nombre de jours dont le débit calculé dépasse ou atteint la capacité, divisé par le nombre de journées exploitables. Les journées incomplètes sont exclues.
- **Rééchantillonnage par blocs :** les blocs historiques se chevauchent et ne sont pas indépendants ; les trajectoires créées sont des scénarios hypothétiques, sans date attribuée dans le futur.
- **Incertitudes C et n :** tirages normaux tronqués respectivement à [0,1] et aux valeurs strictement positives, indépendants l'un de l'autre et de la pluie ; ce choix est pédagogique.
- **Durée :** les cumuls glissants horaires sont divisés par leur nombre d'heures. Cela approxime une intensité sur la durée considérée, pas une courbe IDF ni une période de retour réglementaire.
- **Précision :** la marge affichée est seulement une erreur Monte-Carlo (≈1,96 × √[p(1−p)/N]). L'incertitude de réanalyse, d'échantillonnage climatique et de formule n'y est pas incluse.""")


def teaching_page():
    hero("Comprendre & tester","Exécute les calculs à la main, puis retrouve exactement les mêmes résultats dans l'application.","LABORATOIRE · ENSEIGNEMENT · VÉRIFICATION")
    st.subheader("🧪 Démonstration hors connexion")
    a,b=st.columns(2)
    with a:st.button("Charger l'exemple fictif",on_click=add_demo,type="primary",use_container_width=True)
    with b:st.button("Réinitialiser l'exemple fictif",on_click=reset_demo,use_container_width=True)
    st.markdown("### Test 1 — Débit rationnel")
    st.write("**Chemin :** « 📐 Dimensionnement » → « Hypothèses hydrologiques » → « Recalculer et enregistrer les hypothèses ».")
    st.write("**Saisir :** A = 2 ha, C = 0,65, I = 30 mm/h, durée = 1 h.")
    flow=rational_flow(.65,30,2)
    st.latex(r"Q = \frac{0{,}65\times 30\times 2}{360}=0{,}10833\;m^3/s")
    st.success(f"Résultat attendu : {flow:.5f} m³/s.")
    st.markdown("### Test 2 — Capacité de Manning")
    st.write("**Chemin :** « 📐 Dimensionnement » → « Sections et scénarios » → « Enregistrer les sections ». ")
    st.write("**Saisir :** rectangulaire, b = 0,30 m, H = 0,30 m, z = 0, pente = 0,5 %, n = 0,015, réserve = 0 %.")
    hyd=manning_capacity("Rectangulaire",.3,.3,0,.5,.015,0)
    st.latex(r"A=0{,}3\times0{,}3=0{,}09\;m^2;\quad P=0{,}3+2\times0{,}3=0{,}9\;m")
    st.success(f"Capacité attendue = {hyd['capacite_m3s']:.5f} m³/s. Ratio Q demandé/Q capacité = {flow/hyd['capacite_m3s']:.3f} (dépassement).")
    st.markdown("### Test 3 — Statistique descriptive sans API")
    st.write("**Chemin :** « 🎓 Comprendre et tester » → « Charger l'exemple fictif », puis « 🌧️ Pluviométrie ».")
    st.write("Dix pluies fictives (mm/j) : "+", ".join(map(str,EXAMPLE_DAILY))+".")
    st.write("**Saisir :** durée d = 1 h, seuil de pluie = 10 mm/j.")
    sample=pd.Series(EXAMPLE_DAILY,dtype=float)
    st.success(f"Attendu : moyenne = {sample.mean():.1f} mm/j ; "
               f"jours ≥ 10 mm = {int((sample>=10).sum())} / 10 ; fréquence = {(sample>=10).mean():.0%}.")
    st.markdown("### Test 4 — Simulation reproductible")
    st.write("**Chemin :** « 🎲 Probabilités » → « Simulation de Monte-Carlo par blocs » → « Lancer la simulation ». ")
    st.write("**Saisir :** 3 000 simulations, horizon 10 jours, bloc 3 jours, graine 42, incertitudes C=0 % et n=0 %. Cliquez deux fois avec les mêmes valeurs : les résultats doivent être identiques.")
    st.markdown("### Test 5 — Export et sauvegarde")
    st.write("**Chemin :** « 📐 Dimensionnement » → « Exports complets de l'étude » → « Rapport PDF » ou « Données et résultats Excel ».")
    st.write("**Sauvegarde :** « 📊 Vue générale » → « Rapport et sauvegardes » → « Sauvegarder tous les bassins (JSON) ». Restaurer depuis la barre latérale.")
    st.markdown("### Quelques formules de probabilités")
    st.latex(r"\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i\qquad s^2=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})^2")
    st.latex(r"\widehat{P}(X\ge x_0)=\frac{\#\{X_i\ge x_0\}}{n_{\mathrm{valides}}}")
    st.caption("La variance affichée dans les résultats descriptifs utilise la correction n−1 lorsque n ≥ 2. Les seuils sont des hypothèses personnalisables, pas des normes camerounaises.")


def sidebar_area():
    st.sidebar.markdown("## 🌊 DrainLab **360**")
    st.sidebar.caption("LE LABORATOIRE DU RUISSELLEMENT")
    st.sidebar.divider()
    st.sidebar.radio("Navigation",NAV,key="page",label_visibility="collapsed")
    st.sidebar.divider()
    p=active_project()
    if p:st.sidebar.caption("BASSIN ACTIF");st.sidebar.markdown(f"**{html.escape(p['nom'])}**")
    else:st.sidebar.caption("Aucun bassin sélectionné")
    dark=st.sidebar.toggle("Ambiance sombre",value=False,key="dark_mode")
    st.sidebar.caption("La sauvegarde est manuelle : la session peut être réinitialisée.")
    st.sidebar.download_button("💾 Sauvegarder en JSON",
        export_json(st.session_state.projects,st.session_state.active_id),
        file_name="DrainLab360_sauvegarde.json",mime="application/json",use_container_width=True)
    with st.sidebar.expander("♻️ Restaurer une sauvegarde"):
        restore=st.file_uploader("Fichier DrainLab360 JSON",type=["json"],key="restore_file")
        if restore and st.button("Restaurer les études"):
            try:
                projects,active=restore_json(restore.getvalue())
                st.session_state.projects=projects;st.session_state.active_id=active
                st.session_state.simulation=None
                st.success("Sauvegarde restaurée.");st.rerun()
            except (ValueError,KeyError,TypeError,UnicodeDecodeError,json.JSONDecodeError) as exc:
                st.error(f"Restauration refusée : {exc}")
    st.sidebar.markdown("---")
    st.sidebar.caption("ENSPY · Projet universitaire | Données externes sous conditions des fournisseurs.")
    return dark


def main():
    st.set_page_config(page_title="DrainLab 360 | ENSPY",page_icon="🌊",layout="wide",
        initial_sidebar_state="expanded")
    if "projects" not in st.session_state:st.session_state.projects=[]
    if "active_id" not in st.session_state:st.session_state.active_id=None
    if "page" not in st.session_state:st.session_state.page=NAV[0]
    if "simulation" not in st.session_state:st.session_state.simulation=None
    dark=sidebar_area()
    inject_theme(dark)
    current=st.session_state.page
    if current==NAV[0]:dashboard()
    elif current==NAV[1]:basin_page()
    elif current==NAV[2]:rainfall_page()
    elif current==NAV[3]:design_page()
    elif current==NAV[4]:probability_page()
    elif current==NAV[5]:teaching_page()
    st.divider()
    st.caption("DrainLab 360 • Version pédagogique 1.0 • Les calculs sont des pré-études exploratoires. "
      "Ils n'autorisent ni exécution de travaux ni validation réglementaire.")


if __name__=="__main__":
    main()
