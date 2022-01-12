from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import json
import time
import gc


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
        name = data["name"]
        items = [BenchmarkItem(item) for item in data["items"]]
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
    items: BenchmarkItem

    @staticmethod
    def run(name: str, items: list[Tuple[str, Callable, Callable]]) -> "BenchmarkGroup":
        print("-" * 50)
        print(f"Executing group {name}")
        return BenchmarkGroup(
            name=name, items=[BenchmarkItem.run(*item) for item in items]
        )

    def tojson(self):
        return {
            "name": self.name,
            "items": [item.tojson() for item in self.items],
        }

    @classmethod
    def fromjson(self, data):
        name = data["name"]
        items = [BenchmarkItem(item) for item in data["items"]]
        return BenchmarkGroup(name=name, items=items)


@dataclass
class BenchmarkSet:
    name: str
    groups: list[BenchmarkGroup]

    def write(self, filename: Optional[str] = None):
        if not filename:
            filename = self.name + ".json"
        with open(filename, "w") as f:
            json.dump(self.tojson(), f)

    def tojson(self):
        return {"name": self.name, "groups": [item.tojson() for item in self.groups]}

    @classmethod
    def fromjson(cls, data) -> "BenchmarkSet":
        return BenchmarkSet(
            name=data["name"],
            groups=[BenchmarkGroup.fromjson(item) for item in data["groups"]],
        )

    @staticmethod
    def fromjson_file(filename: str) -> "BenchmarkSet":
        with open(filename, "r") as f:
            return BenchmarkSet.fromjson(json.load(f))
