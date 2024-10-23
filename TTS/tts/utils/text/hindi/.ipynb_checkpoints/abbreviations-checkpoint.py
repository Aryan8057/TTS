import re

# List of (regular expression, replacement) pairs for abbreviations in hindi:
abbreviations_hi = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        # ("mrs", "misess"),
        # ("mr", "mister"),
        ("डॉ.", "डॉक्टर"),
		("रु", "रुपए"),
		("रु.", "रुपए")
		
	
    ]
]
