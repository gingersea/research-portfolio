"""Derive report numbers and compact reference entries from archived evidence."""
import json
from pathlib import Path
import re

here = Path(__file__).resolve().parent
root = here.parents[1]
pilot = json.loads((root / "benchmark/results/training_pilot/metrics.json").read_text())
audit = json.loads((root / "benchmark/results/raw/part_a_summary.json").read_text())["by_model"]
validation = json.loads((root / "benchmark/results/raw/detector_validation.json").read_text())
values = {
    "ParamCount": f"{pilot['parameter_count']:,}",
    "TrainCount": f"{pilot['config']['n_train']:,}",
    "ValCount": str(pilot['config']['n_validation']),
    "TestCount": str(pilot['config']['n_test']),
    "BestEpoch": str(pilot['selected_epoch']),
    "TestIoU": f"{pilot['test']['mean_iou']:.3f}",
    "TestError": f"{pilot['test']['centroid_error_px']:.3f}",
    "BaselineIoU": f"{pilot['color_threshold_test']['mean_iou']:.3f}",
    "BaselineError": f"{pilot['color_threshold_test']['centroid_error_px']:.3f}",
}
for label,epoch in [("First",1),("Third",3),("Sixth",6),("Last",12)]:
    row = next(h for h in pilot["history"] if h["epoch"] == epoch)
    values[f"Loss{label}"] = f"{row['train_loss']:.3f}"
    values[f"ValIoU{label}"] = f"{row['validation']['mean_iou']:.3f}"
for label,model in [("Wan","wan2.1_1.3b"),("Cog","cogvideox_5b")]:
    m = audit[model]
    values[f"{label}Generated"] = str(m["n_videos"])
    values[f"{label}Tracked"] = str(round(m["n_videos"]*m["track_success_rate"]))
    values[f"{label}Fitted"] = str(round(m["n_videos"]*m["fit_success_rate"]))
for label,kind,param in [("FallError","free_fall","g"),("ProjectileError","projectile","g"),("PeriodError","pendulum","T")]:
    r = next(r for r in validation["results"] if r["kind"] == kind and r["param"] == param)
    values[label] = f"{100*r['rel_err']:.3f}"
(here/"numbers.tex").write_text("% Generated from archived experiment JSON; do not hand-edit.\n" +
    "\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in values.items()) + "\n")

# Programmatic metadata extraction produced references.bib. These publication
# enrichments were checked against official CVF/NeurIPS pages and arXiv comments.
venues = {
    "Ho2020denoising": ("NeurIPS", "2020"),
    "Rombach2021high": ("CVPR, pp. 10684--10695", "2022"),
    "Peebles2022scalable": ("ICCV, pp. 4195--4205", "2023"),
    "Yang2024cogvideox": ("ICLR", "2025"),
    "Huang2023vbench": ("CVPR, pp. 21807--21818", "2024"),
    "Kang2024video": ("ICML", "2025"),
    "Kingma2014adam": ("ICLR", "2015"),
    "Motamed2025generative": ("WACV", "2026"),
    "Ronneberger2015convolutional": ("MICCAI, pp. 234--241", "2015"),
}
def tex(value):
    return value.replace("&",r"\&").replace("_",r"\_").replace("%",r"\%")

entries = []
for key,body in re.findall(r"@\w+\{([^,]+),\n(.*?)\n\}", (here/"references.bib").read_text(), re.S):
    fields = dict(re.findall(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$",body,re.M))
    names = fields["author"].split(" and ")
    def short(name):
        if name == "Team Wan":
            return name
        parts = name.split()
        return " ".join(p[0]+"." for p in parts[:-1])+" "+parts[-1]
    author = " and ".join(map(short,names)) if len(names)<=2 else short(names[0])+" et al."
    venue,year = venues.get(key,(fields["howpublished"],fields["year"]))
    punctuation = "'' " if fields["title"].endswith("?") else ",'' "
    entries.append(r"\bibitem{"+key+"} "+tex(author)+", ``"+tex(fields["title"])+punctuation+r"\emph{"+tex(venue)+"}, "+year+".")
assert len(entries) == 13
(here/"references.tex").write_text("% Generated from references.bib with verified venue enrichment.\n"+"\n".join(entries)+"\n")
print(f"Generated {len(values)} numerical macros and {len(entries)} references")
