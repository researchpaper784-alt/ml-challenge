"""Synthetic stand-in for the real dataset (same file layout + noise patterns from the problem PDF).
Used ONLY to smoke-test the pipeline end-to-end. Never used for training the real submission."""
import os
import random
import sys

R = random.Random(7)
WORDS = {
    "US": ["Summit", "Liberty", "Eagle", "Pioneer", "Harbor", "Maple", "Granite", "Cedar", "Atlas", "Beacon",
           "Keystone", "Frontier", "Redwood", "Silver", "Northstar", "Bluewater", "Ironwood", "Lakeside"],
    "India": ["Shri Ganesh", "Lakshmi", "Balaji", "Sai", "Krishna", "Durga", "Om", "Shree Ram", "Mahalaxmi",
              "Annapurna", "Bharat", "Jai Hind", "Ganga", "Venkateshwara", "Shiv Shakti", "Kaveri", "Hanuman"],
    "France": ["Société Générale du", "Boulangerie", "Atelier", "Maison", "Café", "Librairie", "Garage",
               "Pharmacie", "Côté", "Étoile", "Rivière", "Château", "Épicerie"],
}
TRADE = {"US": ["Logistics", "Dental Care", "Auto Repair", "Hardware", "Consulting", "Bakery", "Plumbing"],
         "India": ["Traders", "Textiles", "Enterprises", "Jewellers", "Electricals", "Sweets", "Pharma"],
         "France": ["Martin", "Dupont", "Lefèvre", "Moreau", "Bernard", "Durand", "Petit"]}
SUFFIX = {"US": ["Inc", "LLC", "Corp", "Co", ""], "India": ["Pvt Ltd", "Private Limited", "& Co", "LLP", ""],
          "France": ["SARL", "SAS", "SA", ""]}
STREET = {"US": ["Main St", "Oak Ave", "Elm Street", "Park Blvd", "Washington Rd", "Lake Dr"],
          "India": ["MG Road", "Station Rd", "Nehru Nagar", "Gandhi Chowk", "Ring Road", "Sector 14"],
          "France": ["Rue de Rivoli", "Avenue Victor Hugo", "Bd Saint-Michel", "Rue du Faubourg", "Place Bellecour"]}
CITY = {"US": [("Austin", "TX"), ("Denver", "CO"), ("Boston", "MA"), ("Seattle", "WA")],
        "India": [("Indore", "MP"), ("Pune", "Maharashtra"), ("Jaipur", "Rajasthan"), ("Lucknow", "UP")],
        "France": [("Paris", ""), ("Lyon", ""), ("Marseille", ""), ("Toulouse", "")]}
LANDMARK = {"US": "Near Walmart", "India": "Near SBI ATM", "France": "Près de la gare"}


def entity(c):
    name = f"{R.choice(WORDS[c])} {R.choice(TRADE[c])} {R.choice(SUFFIX[c])}".strip()
    city, st = R.choice(CITY[c])
    num = str(R.randint(1, 999))
    pc = {"US": f"{R.randint(10000, 99999)}", "India": f"{R.randint(110000, 859999)}",
          "France": f"{R.randint(10, 95):02d}{R.randint(0, 999):03d}"}[c]
    if c == "France":
        addr = f"{num} {R.choice(STREET[c])}, {pc} {city}"
    else:
        addr = f"{num}, {R.choice(STREET[c])}, {city}, {st} {pc}"
    return {"name": name, "addr": addr, "country": c}


def typo(s):
    if len(s) < 4:
        return s
    i = R.randrange(1, len(s) - 2)
    op = R.random()
    if op < .33:
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op < .66:
        return s[:i] + s[i + 1:]
    return s[:i] + R.choice("aeinrst") + s[i:]


SWAPS = [("Pvt Ltd", "Private Limited"), ("Private Limited", "Pvt. Ltd."), ("Corp", "Corporation"),
         ("Inc", "Incorporated"), ("& Co", "and Company"), ("Shri", "Sri"), ("Shree", "Sri"),
         ("Lakshmi", "Laxmi"), ("Road", "Rd"), ("Rd", "Road"), ("Street", "St"), ("St", "Street"),
         ("Ave", "Avenue"), ("Nagar", "Ngr"), ("Rue", "R."), ("Société", "Societe"), ("Café", "Cafe")]


def noisy(e):
    n, a = e["name"], e["addr"]
    for x, y in SWAPS:
        if x in n and R.random() < .5:
            n = n.replace(x, y)
        if x in a and R.random() < .5:
            a = a.replace(x, y)
    if R.random() < .3:
        n = typo(n)
    if R.random() < .15:
        w = n.split()
        if len(w) > 2:
            w[0], w[1] = w[1], w[0]
            n = " ".join(w)
    if R.random() < .2:
        n = R.choice([n.upper(), n.lower()])
    parts = [p.strip() for p in a.split(",")]
    if R.random() < .25 and len(parts) > 2:
        parts = parts[:-1]                                    # drop state+postcode
    if R.random() < .25:
        parts.insert(1, LANDMARK[e["country"]])
    if R.random() < .15:
        R.shuffle(parts)
    a = ", ".join(parts)
    if R.random() < .15:
        a = typo(a)
    if R.random() < .03:
        a = ""
    return n, a


def build(split, countries, n_s1, out):
    os.makedirs(out, exist_ok=True)
    s1, s2, s3, gt = [], [], [], []
    ids = {"S2": 0, "S3": 0}

    def new(src, rec, n, a):
        ids[src] += 1
        rid = f"{src}-{ids[src]:05d}"
        rec.append((rid, n, a, e["country"]))
        return rid

    ents = []
    for i in range(n_s1):
        e = entity(R.choice(countries))
        ents.append(e)
        # branch confuser: same name, different address -> a hard negative in the pool
        if R.random() < .15:
            b = dict(e, addr=entity(e["country"])["addr"])
            new(R.choice(["S2", "S3"]), R.choice([s2, s3]), *noisy(b))
    R.shuffle(ents)
    for i, e in enumerate(ents, 1):
        sid = f"S1-{i:05d}"
        s1.append((sid, e["name"], e["addr"], e["country"]))
        k = R.choices([0, 1, 2, 3, 4], [.35, .38, .17, .07, .03])[0]
        m = []
        for _ in range(k):
            src = R.choice(["S2", "S3"])
            m.append(new(src, s2 if src == "S2" else s3, *noisy(e)))
        gt.append((sid, ",".join(m)))
    for _ in range(n_s1 // 2):                                # pool-only distractors
        e = entity(R.choice(countries))
        src = R.choice(["S2", "S3"])
        new(src, s2 if src == "S2" else s3, *noisy(e))
    R.shuffle(s2), R.shuffle(s3)

    def w(fn, rows, hdr):
        with open(os.path.join(out, fn), "w", encoding="utf-8") as f:
            f.write("\t".join(hdr) + "\n")
            for r in rows:
                f.write("\t".join(r) + "\n")
    h = ["entity_id", "business_name", "business_address", "country"]
    w(f"{split}_source1.tsv", s1, h)
    w(f"{split}_source2.tsv", s2, h)
    w(f"{split}_source3.tsv", s3, h)
    w(f"{split}_ground_truth.tsv", gt, ["source1_entity_id", "matched_entity_ids"])


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "synthetic_dataset"
    build("train", ["US", "India"], 3000, os.path.join(root, "train"))
    build("test", ["US", "India", "France"], 1500, os.path.join(root, "test"))
    print("synthetic data written to", root)
