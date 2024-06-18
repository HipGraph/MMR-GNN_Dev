import os

import data.variables as DV


def dataset_name():
    return "solar-energy"


class DataVariables(DV.DataVariables):

    def data_dir(self):
        return os.path.dirname(os.path.realpath(__file__))

    def spatial_label_field(self):
        return "plant"

    def temporal_label_field(self):
        return "date"

    def temporal_label_format(self):
        return "%Y-%m-%d_%H-%M-%S"

    def temporal_seasonality_period(self):
        return [1, "days"]

    def temporal_resolution(self):
        return [10, "minutes"]

    def temporal_partition(self):
        selections = [
            ["ordered-split", 0.0, 0.6], 
            ["ordered-split", 0.6, 0.8], 
            ["ordered-split", 0.8, 1.0], 
        ]
        return selections, self.partitions()


class SpatiotemporalDataVariables(DataVariables, DV.SpatiotemporalDataVariables):

    def feature_fields(self):
        return ["date", "power_MW"]

    def shape(self):
        return ["temporal", "spatial", "feature"]
