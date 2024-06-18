import os

import data.variables as DV


def dataset_name():
    return "metr-la"


class DataVariables(DV.DataVariables):

    def data_dir(self):
        return os.path.dirname(os.path.realpath(__file__))

    def spatial_label_field(self):
        return "sensor"

    def node_label_field(self):
        return "sensor"

    def temporal_label_field(self):
        return "date"

    def temporal_label_format(self):
        return "%Y-%m-%d_%H-%M-%S"

    def temporal_seasonality_period(self):
        return [1, "days"]

    def temporal_resolution(self):
        return [5, "minutes"]

    def temporal_partition(self):
        selections = [
            ["ordered-split", 0.0, 0.7], 
            ["ordered-split", 0.7, 0.9], 
            ["ordered-split", 0.9, 1.0], 
        ]
        return selections, self.partitions()


class SpatialDataVariables(DataVariables, DV.SpatialDataVariables):

    def feature_fields(self):
        return ["latitude", "longitude"]


class SpatiotemporalDataVariables(DataVariables, DV.SpatiotemporalDataVariables):

    def feature_fields(self):
        return ["date", "speedmph"]

    def shape(self):
        return ["temporal", "spatial", "feature"]


class GraphDataVariables(DataVariables, DV.GraphDataVariables):

    def edge_source_field(self):
        return "source"

    def edge_destination_field(self):
        return "destination"

    def edge_weight_field(self):
        return "weight"
