import sys
from benchmark import BenchmarkSet, BenchmarkItemAggregate
import glob
import json
from matplotlib.figure import Figure
import os
import webbrowser
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('SVG')


def load_report(filename: str) -> BenchmarkSet:
    print(f"Loading {filename}")
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
    benchmarks.sort(key=lambda x: x.name + '\b' + x.environment)
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
<!-- Required meta tags -->
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Bootstrap CSS -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-1BmE4kWBq78iYhFldvKuhfTAU6auU8tT94WrHftjDbrCEXSU1oBoqyl2QvZ6jIW3" crossorigin="anonymous">
<title>Tensor C++ library benchmarks</title>
</head>
<body>
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-ka7Sk0Gln4gmtz2MlQnikT1wXgYsOg+OMhuP+IlRH9sENBO0LRn5q+8nbTov4+1p" crossorigin="anonymous"></script>
<script src="http://code.jquery.com/jquery-latest.min.js"></script>
<script type="text/javascript" >
    $(document).on('click', 'a[href^="#"]', function (event) {
    event.preventDefault();

    $('html, body').animate({
        scrollTop: $($.attr(this, 'href')).offset().top
    }, 200);
    });
</script>
"""

HTML_REPORT_FOOTER = "</body>\n</html>"


def html_write_headers(f, benchmarks):
    f.write(HTML_REPORT_HEADER)


def html_group_selector(f, group_item_pairs):
    groups = set([group for group, _ in group_item_pairs])
    dropdown = {g: '' for g in groups}
    output = []
    for n, (group, item) in enumerate(group_item_pairs):
        label = f'benchmark{n}'
        button = f'<li><a class="dropdown-item" href="#{label}">{item}</a></li>\n'
        dropdown[group] += button
        output.append((group, item, label))
    f.write('<div style="padding:1em; background:#2e2929" class="container-fluid border-bottom shadow text-center dropdown sticky-top d-grid gap-2 d-md-block">')
    for n, key in enumerate(sorted(dropdown)):
        options = dropdown[key]
        button_label = f'dropdownMenuButton{n}'
        f.write(f'<a class="btn btn-sm btn-secondary dropdown-toggle" type="button"'
                f'id="{button_label}" href="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">'
                f'{key}</a>')
        f.write(
            f'<ul class="dropdown-menu" aria-labelledby="{button_label}">\n')
        f.write(options)
        f.write('</ul>\n')
    f.write('</div>\n')
    return output


def html_write_footer(f, benchmarks):
    f.write("<div class='container-fluid border-top fixed-bottom small shadow' style='background:#ede7e7;padding-top:1em'>")
    f.write('<footer class="container">')
    f.write("<ul>")
    for b in benchmarks:
        f.write(f"<li>{b.name} - {b.environment}</li>")
    f.write("</ul>")
    f.write('</footer>')
    f.write("</div>")
    f.write(HTML_REPORT_FOOTER)


def produce_svg_image(agg, imagefile):
    fig = plot_aggregate(agg)
    fig.savefig(imagefile)
    plt.close(fig)


def html_display_plot_with_caption(f, group_name, item_name, label, imageuri):
    f.write('<div class="row">\n')
    f.write('<figure class="figure">\n')
    f.write(
        f'<img id="{label}" class="img-fluid text-center" src="{imageuri}" alt="{label}">\n')
    f.write(
        f'<figcaption class="figure-caption">{group_name} / {item_name}</figcaption>\n')
    f.write('</figure>\n')
    f.write('</div>\n')


def html_report(filename: str, benchmarks: list[BenchmarkSet], browse: bool = True):
    if filename[-5:] != ".html":
        raise Exception(f'HTML file name "{filename}" lacks .html extension')
    benchmarks = disambiguate_benchmarks(benchmarks)
    imagedir = filename[:-5] + "/"
    if not os.path.exists(imagedir):
        os.makedirs(imagedir)
    n = 0
    with open(filename, "w") as f:
        html_write_headers(f, benchmarks)
        group_item_pairs = html_group_selector(f,
                                               BenchmarkSet.find_all_pairs(benchmarks))
        f.write('<div class="container" style="padding-top:1em">')
        f.write("<h1>Benchmark report</h1>")
        last_group = ''
        for group_name, item_name, label in group_item_pairs:
            imagefile = f"{imagedir}/figure-{n}.svg"
            imageuri = f"{os.path.basename(filename[:-5])}/figure-{n}.svg"
            agg = BenchmarkItemAggregate(benchmarks, group_name, item_name)
            produce_svg_image(agg, imagefile)
            if group_name != last_group:
                f.write(f'<h2>{group_name}</h2>')
                last_group = group_name
            html_display_plot_with_caption(
                f, group_name, item_name, label, imageuri)
            n = n + 1
        f.write('</div>')
        html_write_footer(f, benchmarks)
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
