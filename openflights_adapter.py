# -*- coding: utf-8 -*-
"""
Lightweight adapter for OpenFlights data files.
Parses airports.dat (or airports-extended.dat) and routes.dat if present.

Notes on formats (from OpenFlights):
- airports.dat columns: Airport ID, Name, City, Country, IATA, ICAO, Latitude, Longitude,
  Altitude, Timezone, DST, Tz database time zone, Type, Source. Values may be quoted; null as \\N.
- routes.dat columns: Airline, Airline ID, Source airport, Source airport ID,
  Destination airport, Destination airport ID, Codeshare, Stops, Equipment
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional, Tuple


def _as_float(v: str, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_str(v: str) -> str:
    v = (v or '').strip()
    return '' if v == "\\N" else v


def load_airports(file_path: str) -> List[Dict]:
    airports: List[Dict] = []
    if not file_path or not os.path.exists(file_path):
        return airports
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        # OpenFlights uses CSV with optional quotes
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 7:
                continue
            try:
                ap = {
                    'airport_id': _as_str(row[0]) if len(row) > 0 else '',
                    'name': _as_str(row[1]) if len(row) > 1 else '',
                    'city': _as_str(row[2]) if len(row) > 2 else '',
                    'country': _as_str(row[3]) if len(row) > 3 else '',
                    'iata': _as_str(row[4]) if len(row) > 4 else '',
                    'icao': _as_str(row[5]) if len(row) > 5 else '',
                    'lat': _as_float(row[6]) if len(row) > 6 else 0.0,
                    'lng': _as_float(row[7]) if len(row) > 7 else 0.0,
                    'tz': _as_str(row[11]) if len(row) > 11 else '',
                    'type': _as_str(row[12]) if len(row) > 12 else 'airport',
                    'source': _as_str(row[13]) if len(row) > 13 else '',
                }
                # Only keep plausible coordinates
                if -90.0 <= ap['lat'] <= 90.0 and -180.0 <= ap['lng'] <= 180.0:
                    airports.append(ap)
            except Exception:
                # Skip malformed row
                continue
    return airports


def load_airports_dafif(file_path: str) -> List[Dict]:
    """Parse DAFIF-style airport list where columns are:
    CC,Name,ICAO,IATA,lon,lat,elev

    Returns list of airports with fields compatible to load_airports output.
    """
    airports: List[Dict] = []
    if not file_path or not os.path.exists(file_path):
        return airports
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            try:
                # DAFIF: [0]=CC, [1]=Name, [2]=ICAO, [3]=IATA, [4]=lon, [5]=lat
                name = _as_str(row[1]) if len(row) > 1 else ''
                icao = _as_str(row[2]).upper() if len(row) > 2 else ''
                iata = _as_str(row[3]).upper() if len(row) > 3 else ''
                lon = _as_float(row[4]) if len(row) > 4 else 0.0
                lat = _as_float(row[5]) if len(row) > 5 else 0.0
                ap = {
                    'airport_id': '',
                    'name': name,
                    'city': '',
                    'country': _as_str(row[0]) if len(row) > 0 else '',  # country code best-effort
                    'iata': iata,
                    'icao': icao,
                    'lat': lat,
                    'lng': lon,
                    'tz': '',
                    'type': 'airport',
                    'source': 'dafif'
                }
                if -90.0 <= ap['lat'] <= 90.0 and -180.0 <= ap['lng'] <= 180.0:
                    airports.append(ap)
            except Exception:
                continue
    return airports


def load_routes(file_path: str) -> Dict[str, Dict[str, int]]:
    """Return mapping src_iata -> {dst_iata: count}
    Counts occurrences to give rough frequency weights if duplicates exist.
    """
    routes: Dict[str, Dict[str, int]] = {}
    if not file_path or not os.path.exists(file_path):
        return routes
    with open(file_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            try:
                src = _as_str(row[2]).upper()
                dst = _as_str(row[4]).upper()
                if not src or not dst or src == '\\N' or dst == '\\N':
                    continue
                routes.setdefault(src, {})[dst] = routes.setdefault(src, {}).get(dst, 0) + 1
            except Exception:
                continue
    return routes


def build_indexes(airports: List[Dict]) -> Dict:
    by_iata: Dict[str, Dict] = {}
    by_icao: Dict[str, Dict] = {}
    for ap in airports:
        iata = (ap.get('iata') or '').upper()
        icao = (ap.get('icao') or '').upper()
        if iata:
            by_iata[iata] = ap
        if icao:
            by_icao[icao] = ap
    return {'airports': airports, 'by_iata': by_iata, 'by_icao': by_icao}


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    import math
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


def nearest_airports(index: Dict, lat: float, lng: float, k: int = 5, airport_only: bool = True, require_iata: bool = True) -> List[Dict]:
    airports = index.get('airports') or []
    scored: List[Tuple[float, Dict]] = []
    for ap in airports:
        if airport_only and (ap.get('type') and ap.get('type') != 'airport'):
            continue
        if require_iata and not ap.get('iata'):
            continue
        try:
            d = _haversine_km(lat, lng, float(ap['lat']), float(ap['lng']))
            scored.append((d, ap))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0])
    out: List[Dict] = []
    for dist, ap in scored[:max(1, int(k))]:
        rec = {
            'airport_id': ap.get('airport_id'),
            'name': ap.get('name'),
            'city': ap.get('city'),
            'country': ap.get('country'),
            'iata': ap.get('iata'),
            'icao': ap.get('icao'),
            'lat': ap.get('lat'),
            'lng': ap.get('lng'),
            'distance_km': round(float(dist), 3),
        }
        out.append(rec)
    return out
