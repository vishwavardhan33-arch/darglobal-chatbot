# Verify the seeds in a browser first; adjust include/skip patterns as needed.
SITES = {
    "darglobal": {
        "name": "DAR Global",
        "seeds": ["https://www.darglobal.co.uk/"],
        "hosts": {"darglobal.co.uk", "www.darglobal.co.uk"},
        "max_pages": 60, "max_depth": 3,
        "skip_regex": r"/(login|signin|privacy|cookie|terms)",
    },
    "wasalt": {
        "name": "Wasalt",
        "seeds": ["https://wasalt.sa/"],
        "hosts": {"wasalt.sa", "www.wasalt.sa"},
        "max_pages": 80, "max_depth": 2,
        "skip_regex": r"^/ar(/|$)|/(login|signin|signup|privacy|cookie|terms)",
    },
}
