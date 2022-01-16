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


def plot_aggregate(agg: BenchmarkItemAggregate, doprint: bool = False) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, y in zip(agg.columns, agg.times):
        if doprint:
            print(label)
            print(y)
        ax.plot(agg.sizes, y, label=label)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylabel("time (s)")
    ax.set_xlabel("size")
    ax.legend()
    fig.tight_layout()
    return fig


def disambiguate_benchmarks(benchmarks: list[BenchmarkSet]) -> list[BenchmarkSet]:
    benchmarks.sort(key=lambda x: x.name + x.environment)
    names = {}
    for b in benchmarks:
        names[b.name] = names.get(b.name, []) + [b]
    for name, set in names.items():
        if len(set) > 1:
            for n, b in enumerate(set):
                b.name = f"{name} ({n})"
    return benchmarks


HTML_REPORT_HEADER = """<html>
<head>
</head>
<body>"""

HTML_REPORT_FOOTER = "</body>\n</html>"


def html_report(filename: str, benchmarks: list[BenchmarkSet], browse: bool = True):
    if filename[-5:] != ".html":
        raise Exception(f'HTML file name "{filename}" lacks .html extension')
    benchmarks = disambiguate_benchmarks(benchmarks)
    imagedir = filename[:-5] + "/"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    n = 0
    with open(filename, "w") as f:
        f.write(HTML_REPORT_HEADER)
        f.write("<h1>Benchmark report</h1>")
        f.write("<ul>")
        for b in benchmarks:
            f.write(f"<li>{b.name} - {b.environment}</li>")
        f.write("</ul>")
        for group_name, item_name in BenchmarkSet.find_all_pairs(benchmarks):
            agg = BenchmarkItemAggregate(benchmarks, group_name, item_name)
            f.write(f"<h2>Benchmark {group_name}.{item_name}</h2>\n")

            imagefile = f"{imagedir}/figure-{n}.svg"
            imageuri = f"{os.path.basename(filename[:-5])}/figure-{n}.svg"
            fig = plot_aggregate(
                agg, doprint=(group_name == "RTensor" and item_name == "plus")
            )
            fig.savefig(imagefile)
            plt.close(fig)
            f.write(f'<img src="{imageuri}">')
            n = n + 1
        f.write(HTML_REPORT_FOOTER)
    if browse:
        filename = os.path.abspath(filename)
        webbrowser.open("file:///" + filename, autoraise=True)


def main(argv: list[str]):
    if len(argv) < 2:
        root = "./profile/"
    else:
        root = argv[1]
    if len(argv) < 3:
        html_file = root + "/report.html"
    else:
        html_file = argv[2]
    files = glob.glob(root + "/*.json")
    data = [load_report(f) for f in files]
    html_report(html_file, data)


if __name__ == "__main__":
    main(sys.argv)
