from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Protocol, List, Tuple
import numpy as np
import math

@dataclass
class LPBFInterface(ABC):
    """
    - Context info: dict
                - Cube position (the n-th cube)
                - Layer i, can infer the height in z-direction
                - Line i
                - Laser power
                - start coord
                - end coord
                - Scaning Speed (remain constant in this case)
                - Scanning vector (will infer from coord)
                - Regime info (4 - class classifications)
                    - Base | Base plate, ignored
                    - GP   | Gas pore, or keyhole 
                    - NP   | No pore
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion

            - Defect labels: list[str]
                    - NP   | No pore
                    - GP   | Gas pore, or keyhole 
                    - RLoF | Random lack of fusion
                    - LoF  | Lack of fusion
            
            - In-process data: dict
                - microphone
                - AE
                - photodiode
    """
    context_info: dict
    defect_labels: dict
    in_process_data: dict

    @property
    def cube_position(self) -> list[int]:
        return self.context_info["cube_position"]

    @property
    def layer_i(self) -> list[int]:
        return self.context_info["layer_i"]

    @property
    def line_i(self) -> list[int]:
        return self.context_info["line_i"]

    @property
    def laser_power(self) -> list[float]:
        return self.context_info["laser_power"]

    @property
    def start_coord(self) -> list[float]:
        return self.context_info["start_coord"]

    @property
    def end_coord(self) -> list[float]:
        return self.context_info["end_coord"]

    @property
    def scanning_speed(self) -> list[float]:
        return self.context_info["scanning_speed"]

    @property
    def _calculate_vector(self) -> list[str]:
        return self._get_direction

    @property
    def regime_info(self) -> dict[str]:
        return self.context_info["regime_info"]

    @property
    def microphone(self) -> list[np.asarray]:
        return self.in_process_data["microphone"]

    @property
    def AE(self) -> list[np.asarray]:
        return self.in_process_data["AE"]

    @property
    def photodiode(self) -> list[np.asarray]:
        return self.in_process_data["photodiode"]

    @property
    def _get_direction(self) -> tuple[(float,float)]:
        import dask.array as da
        """
        Infer the direction and magnitude (the speed) from coordinate
        """
        start_coords_dask = da.from_array(self.context_info["start_coord"])
        end_coords_dask = da.from_array(self.context_info["end_coord"])
        delta_coords_dask = end_coords_dask - start_coords_dask
        distances_dask = np.linalg.norm(delta_coords_dask, axis=1)
        degrees_dask = np.degrees(np.arctan2(delta_coords_dask[:, 1], delta_coords_dask[:, 0]))

        distances, directions = distances_dask.compute(), degrees_dask.compute()

        return distances, directions
        # # if magnitude > 0:
        # #     direction /= magnitude
        # return degree, magnitude

    @property
    def _construct_dataset(self,context_info,defect_labels,in_process_data):
        self.context_info = context_info
        self.defect_labels = defect_labels
        self.in_process_data = in_process_data

    @property
    def point_labels(self) -> list[np.asarray]:
        return self.defect_labels["point_labels"]

    @property
    def line_labels(self) -> list[np.asarray]:
        return self.defect_labels["line_labels"]

class LPBFData(LPBFInterface):
    def __init__(self,context_info,in_process_data, defect_labels,):
        self.context_info = context_info
        self.in_process_data = in_process_data
        self.defect_labels = defect_labels
        self.print_vector = self._calculate_vector
        self._get_id()
        self.len = len(self.ids)
    
    # def _get_len(self):
        # def count_elements(lst):
        #     count = 0
        #     for item in lst:
        #         if isinstance(item, list):
        #             count += count_elements(item)
        #         else:
        #             count += 1
        #     return count 
        # return count_elements(self.point_labels)

    def _get_id(self):
        ids = []
        _previous_line_idx = -1
        for item_id, _line_idx in enumerate(self.line_i):
            _layer_idx = self.layer_i[item_id]
            _current_point_labels = self.point_labels[item_id]
            for _point_idx, _point_label in enumerate(_current_point_labels):
                daq_idx = int((_point_idx+1)/len(_current_point_labels)*len(self.AE[_line_idx]))
                ids.append([item_id, _layer_idx, _line_idx, daq_idx, _point_label])
        self.ids = np.asarray(ids,dtype=int)

class LPBFPointData(LPBFInterface):
    def __init__(self,context_info,defect_labels,in_process_data):
        self.context_info = context_info
        self.defect_labels = defect_labels
        self.in_process_data = in_process_data
        self.print_vector = self._calculate_vector

if __name__ =="__main__":
    lpbf_data = LPBFData(
    context_info={
        "cube_position": [1,2,3,4],
        "laser_power": [100,200,300,400],
        "start_coord": [(-200,100),( 200,100),( 100,-100),( 100,100)],
        "end_coord":   [( 200,100),(-200,100),(-100, 100),(-100,-100)],
        "scanning_speed": [10,20,30,40],
        "regime_info": ["GP","NP","RLoF","LoF"],
    },
    defect_labels=["GP", "NP","RLoF","LoF"],
    in_process_data={
        "microphone": [
            [0.1,0.2],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3,0.4],
            [0.1,0.2,0.3,0.4,0.5],
            ],
        "AE": [
            [0.1,0.2],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3,0.4],
            [0.1,0.2,0.3,0.4,0.5],
            ],
        "photodiode": [
            [0.1,0.2],
            [0.1,0.2,0.3],
            [0.1,0.2,0.3,0.4],
            [0.1,0.2,0.3,0.4,0.5],
            ],
        }
    )

    print(f"Position:\n{lpbf_data.cube_position}")  
    print(f"Regime info:\n{lpbf_data.regime_info}")    
    print(f"microphone:\n{lpbf_data.microphone}")     
    # print(f"scaning distances:\n{lpbf_data._calculate_vector[0]}") 
    # print(f"scaning directions:\n{lpbf_data._calculate_vector[1]}") 
    print(f"scaning vector:\n{lpbf_data.print_vector}") 
