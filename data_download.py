import urllib.request, json 
import sys

SUBJECT = sys.argv[1]
LINK = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles=%s&prop=extracts&explaintext"

with urllib.request.urlopen(LINK % SUBJECT) as url:
    data = json.loads(url.read().decode())
    
with open("data/%s.txt" % SUBJECT, "w") as datafile:
    val = list(data["query"]["pages"].values())[0]["extract"]
    datafile.write(val)