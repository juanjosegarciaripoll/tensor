from dataclasses import dataclass
from typing import Callable, Optional, Tuple, List
import json
import time
import gc
import warnings
import numpy as np


@dataclass
class BenchmarkItem:
    """Benchmark measurements for one operation.

    This class records the measurements of execution times for different problem sizes.
    """

    name: str
    sizes: list[int]
    times: list[float]

    def tojson(self):
        return {
            "name": self.name,
            "sizes": self.sizes,
            "times": self.times,
        }

    @classmethod
    def fromjson(self, data):
        return BenchmarkItem(
            name=data["name"], sizes=data["sizes"], times=data["times"]
        )

    @classmethod
    def timeit(cls, function: Callable, number: int):
        """Execute `function` a `number` of times and return time taken in seconds."""
        gcold = gc.isenabled()
        gc.disable()
        try:
            t = time.perf_counter()
            for _ in range(number):
                function()
            t = time.perf_counter() - t
        finally:
            if gcold:
                gc.enable()
        return t

    @classmethod
    def autorange(self, function: Callable, limit: float = 0.2):
        i: int = 1
        while True:
            for j in 1, 2, 5:
                number = i * j
                time_taken = self.timeit(function, number)
                if time_taken >= limit:
                    return time_taken / number
            i *= 10

    @staticmethod
    def run(
        name: str,
        function: Callable,
        setup: Optional[Callable[[int], tuple]] = None,
        sizes: list[int] = None,
        limit: float = 0.2,
    ):
        if sizes == None:
            sizes = [4 ** i for i in range(0, 12)]
        times = []
        for s in sizes:
            args = setup(s)
            timing = BenchmarkItem.autorange(lambda: function(*args), limit)
            times.append(timing)
            print(f"Executing item {name} at size {s} took {timing:5g} seconds")
        return BenchmarkItem(name=name, sizes=sizes, times=times)


@dataclass
class BenchmarkGroup:
    name: str
    items: list[BenchmarkItem]

    @staticmethod
    def run(name: str, items: list[Tuple[str, Callable, Callable]]) -> "BenchmarkGroup":
        print("-" * 50)
        print(f"Executing group {name}")
        return BenchmarkGroup(
            name=name, items=[BenchmarkItem.run(*item) for item in items]
        )

    def find_item(self, name: str, error: bool = False):
        for it in self.items:
            if it.name == name:
                return it
        message = f"Item {name} missing from group {self.name}"
        if error:
            raise Exception(message)
        return None

    def tojson(self):
        return {
            "name": self.name,
            "items": [item.tojson() for item in self.items],
        }

    @classmethod
    def fromjson(self, data):
        name = data["name"]
        items = [BenchmarkItem.fromjson(item) for item in data["items"]]
        return BenchmarkGroup(name=name, items=items)


@dataclass
class BenchmarkSet:
    name: str
    groups: list[BenchmarkGroup]
    environment: str

    def write(self, filename: Optional[str] = None):
        if not filename:
            filename = self.name + ".json"
        with open(filename, "w") as f:
            json.dump(self.tojson(), f)

    def tojson(self):
        return {
            "name": self.name,
            "environment": self.environment,
            "groups": [item.tojson() for item in self.groups],
        }

    def find_group(self, name: str, error: bool = False):
        for g in self.groups:
            if g.name == name:
                return g
        message = f"Group {name} missing from benchmark {self.name}"
        if error:
            raise Exception(message)
        return None

    @classmethod
    def fromjson(cls, data) -> "BenchmarkSet":
        return BenchmarkSet(
            name=data["name"],
            environment=data["environment"],
            groups=[BenchmarkGroup.fromjson(item) for item in data["groups"]],
        )

    @staticmethod
    def fromjson_file(filename: str) -> "BenchmarkSet":
        with open(filename, "r") as f:
            return BenchmarkSet.fromjson(json.load(f))

    @staticmethod
    def find_all_pairs(benchmarks: List["BenchmarkSet"]) -> List[Tuple[str, str]]:
        output = set()
        for b in benchmarks:
            for g in b.groups:
                for i in g.items:
                    output.add((g.name, i.name))
        output = list(output)
        output.sort(key=lambda p: ".".join(p))
        return output


@dataclass
class BenchmarkItemAggregate:

    columns: list[str]
    sizes: list[int]
    times: np.ndarray = np.zeros((0, 0))

    def __init__(self, benchmarks: list[BenchmarkSet], group_name: str, item_name: str):
        if not benchmarks:
            return
        items = []
        valid = []
        for set in benchmarks:
            item = None
            group = set.find_group(group_name, None)
            if group:
                item = group.find_item(item_name, None)
            if item:
                items.append(item)
                valid.append(set)
            else:
                warnings.warn(
                    f"Benchmark set {set.name} lacks group {group_name} or item {item_name}"
                )
        self.columns = [b.name for b in valid]
        self.sizes = items[0].sizes
        for n, set in enumerate(valid):
            if not np.all(items[n].sizes == self.sizes):
                raise Exception(
                    f"Benchmark set {set.name} has differring sizes for group {group_name} and item {item_name}"
                )
        self.times = np.array([i.times for i in items])
        if group_name == "RTensor" and item_name == "plus":
            for set in benchmarks:
                print(set.name)
                item = set.find_group(group_name).find_item(item_name)
                print(item.times)
            print(self.columns)
            print(self.times)
