"""
field_config.py
---------------
Patterns and units configuration for Fresenius 4008S Dialysis Machine display
(with fuzzy OCR label aliases).
"""

# Patterns and units for Fresenius 4008S Dialysis Machine display (with fuzzy OCR label aliases)
FIELD_CONFIG = {
    "UF Volume":        {"regex": r"(UF|LF|UV|UN|Diansrs|Dialysis)?\s*(Volume|Vol|Volun|Voi|Vot)", "unit": "ml"},
    "UF Time Left":     {"regex": r"(UF|LF|UV|UN)?\s*Tim[ea]\s*(Left|Lot|Led|Let|Lft|Lel)?",        "unit": "h:min"},
    "UF Rate":          {"regex": r"(UF|LF|UV|UN)?\s*(Rate|Rale|Ral|Rte)",                         "unit": "ml/h"},
    "UF Goal":          {"regex": r"(UF|LF|UV|UN)?\s*(Goal|God|Goa|Gol)",                          "unit": "ml"},
    "Eff. Blood Flow":  {"regex": r"(Eff\.?|Bff|Bid|E\")?\s*B[io]{1,2}d?\s*(Flow|Flot|Fiot|Flo)",   "unit": "ml/min"},
    "Cum. Blood Vol.":  {"regex": r"(Cum\.?|Cun|Cumn|Hood|Blood|Blod)?\s*(Blood|Blod|Daadia)?\s*(Vol|Voi)", "unit": "l"},
    "Kt/V":             {"regex": r"Kt\s*/?\s*V|KI\s*/?\s*V|K1\s*/?\s*V|Ktv",                      "unit": ""},
    "Plasma Na":        {"regex": r"Plasma\s*(Na|N|Na\+)?|FlemNa|PlemNa|PlamNa|Pheni|phenu",       "unit": "mmol/l"},
    "Goal in":          {"regex": r"Goal\s*in|Gol\s*in",                                           "unit": "h:min"},
    "Clearance":        {"regex": r"Clearance|Claance|Charanco|Cledance|Cladance",                 "unit": "ml/min"},
}
