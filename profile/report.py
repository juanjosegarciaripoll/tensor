import os
import webbrowser
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import json
import glob
from benchmark import BenchmarkSet, BenchmarkItemAggregate
import sys


def load_report(filename: str) -> BenchmarkSet:
    with open(filename, "r") as f:
        data = json.load(f)
    return BenchmarkSet.fromjson(data)


def plot_aggregate(agg: BenchmarkItemAggregate) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, y in zip(agg.columns, agg.times):
        ax.plot(agg.sizes, y, label=label)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("time (s)")
    ax.set_xlabel("size")
    ax.legend()
    fig.tight_layout()
    return fig


def html_report(filename: str, benchmarks: list[BenchmarkSet], browse: bool = True):
    if filename[-5:] != ".html":
        raise Exception(f'HTML file name "{filename}" lacks .html extension')
    imagedir = filename[:-5] + "/"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    n = 0
    with open(filename, "w") as f:
        f.write("<h1>Benchmark report</h1>")
        for group in benchmarks[0].groups:
            for item in group.items:
                agg = BenchmarkItemAggregate(benchmarks, group.name, item.name)
                f.write(f"<h2>Benchmark {group.name}.{item.name}</h2>\n")

                imagefile = f"{imagedir}/figure-{n}.svg"
                imageuri = f"{os.path.basename(filename[:-5])}/figure-{n}.svg"
                fig = plot_aggregate(agg)
                fig.savefig(imagefile)
                f.write(f'<img src="{imageuri}">')
    if browse:
        webbrowser.get("windows-default").open(filename, 1)


def main(argv: list[str]):
    if len(argv) < 2:
        root = "./profile/"
    else:
        root = argv[1]
    if len(argv) < 3:
        html_file = "./profile/report.html"
    else:
        html_file = argv[2]
    files = glob.glob(root + "/*.json")
    data = [load_report(f) for f in files]
    html_report(html_file, data)


if __name__ == "__main__":
    main(sys.argv)
