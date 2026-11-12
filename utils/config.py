from typing import Dict, List

# Consistent color palettes
COLORS: Dict[str, str] = {
    "Died": "#EF553B",       # Coral Red
    "Survived": "#00CC96",   # Teal Green
    "First": "#636EFA",
    "Second": "#AB63FA",
    "Third": "#FFA15A",
}

# Explicit category ordering for Plotly charts
ORDERS: Dict[str, List[str]] = {
    "class": ["First", "Second", "Third"],
    "survival": ["Survived", "Died"],
    "age_group": ["child", "teen", "young adult", "adult", "senior"],
    "port": ["Southampton", "Cherbourg", "Queenstown"],
}

# Standard column display labels
LABELS: Dict[str, str] = {
    "family_size": "Family Size (incl. passenger)",
    "age": "Age (Years)",
    "survival": "Outcome",
    "class": "Passenger Class",
    "port": "Embarkation Port",
    "age_group": "Age Group",
}