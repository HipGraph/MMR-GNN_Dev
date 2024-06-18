import warnings
import pandas as pd
import os
import shutil
import glob
import numpy as np
import datetime as dt
import h5py
from inspect import currentframe
from pathlib import Path

from data.caltrans_pems.pems.handler import PeMSHandler

import util
from container import Container
from data.selection import DataSelection
from data.graph import A_to_edgelist, W_to_edgelist


def data_dir():
    return os.sep.join(os.path.realpath(__file__).replace(".py", "").split(os.sep)[:-1])


def download_dir():
    return os.sep.join([str(Path.home()), "Downloads"])


class Integrator:

    debug = 2

    def __init__(self):
        os.makedirs(self.root_dir(), exist_ok=True)
        os.makedirs(self.cache_dir(), exist_ok=True)
        os.makedirs(self.acquire_dir(), exist_ok=True)
        os.makedirs(self.convert_dir(), exist_ok=True)

    def name(self):
        return self.__class__.__name__

    def root_dir(self):
        return os.sep.join([data_dir(), self.name()]) 

    def cache_dir(self):
        return os.sep.join([self.root_dir(), "Integration", "Cache"])

    def acquire_dir(self):
        return os.sep.join([self.root_dir(), "Integration", "Acquired"])

    def convert_dir(self):
#        return os.sep.join([self.root_dir(), "Integration", "Converted"])
        return self.root_dir()

    def spatial_labels_inpath(self):
        return None

    def temporal_labels_inpath(self):
        return None

    def spatial_features_inpath(self):
        return None

    def temporal_features_inpath(self):
        return None

    def spatiotemporal_features_inpath(self):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def graph_features_inpath(self):
        return None

    def spatial_labels_fname(self):
        return "SpatialLabels.csv"

    def temporal_labels_fname(self):
        return "TemporalLabels.csv"

    def spatial_features_fname(self):
        return "Spatial.csv"

    def temporal_features_fname(self):
        return "Temporal.csv"

    def spatiotemporal_features_fname(self):
        return "Spatiotemporal.csv.gz"

    def graph_features_fname(self):
        return "Graph.csv"

    def spatial_labels_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatial_labels_fname()])

    def temporal_labels_outpath(self):
        return os.sep.join([self.convert_dir(), self.temporal_labels_fname()])

    def spatial_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatial_features_fname()])

    def temporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.temporal_features_fname()])

    def spatiotemporal_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.spatiotemporal_features_fname()])

    def graph_features_outpath(self):
        return os.sep.join([self.convert_dir(), self.graph_features_fname()])

    def acquire(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))

    def convert(self, args):
        func_name = "%s.%s" % (self.__class__.__name__, currentframe().f_code.co_name)
        raise NotImplementedError("Implement %s() please!" % (func_name))


class Conversion:

    def dataframe_insert(df_a, df_b, on):
        if not isinstance(on, str):
            raise ValueError("Input \"on\" must be a str. Received %s" % (type(on)))
        elif not (on in df_a.columns and on in df_b.columns):
            raise ValueError("Input \"on\" must be a column of df_a and df_b")
        mask_ab = df_a[on].isin(df_b[on])
        mask_ba = df_b[on].isin(df_a[on])
        df_a.loc[mask_ab] = df_b.loc[mask_ba].set_index(mask_ab[mask_ab].index)
        return df_a


class Electricity(Integrator):

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "electricity.txt.gz"])

    def acquire(self, args):
        print(
            "Download \"electricityloaddiagrams20112014.zip\" from \"https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014\" and unzip into acquisition directory \"%s\"" % (self.acquire_dir())
        )

    def convert(self, args):
        path = os.sep.join([self.acquire_dir(), "LD2011_2014.txt", "LD2011_2014.txt"])
        df = pd.read_csv(path, sep=";", decimal=",")
        if args.debug:
            print(df)
        spatial_labels = np.array(list(df.columns[1:]))
        temporal_labels = np.array(util.generate_temporal_labels("2012-01-01_00-00-00", 24*365*3, [1, "hours"]))
        temporal_labels = np.array([tmp.replace(" ", "_").replace(":", "-") for tmp in df.iloc[:,0]])
        features = df[df.columns[1:]].to_numpy()
        # Clearn by removing the first year (contains all zeros) and down-sampling to kWh (f / 4)
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        start = 4 * 24 * 365 
        temporal_labels = temporal_labels[start:]
        features = features[start:,:]
        n_temporal = len(temporal_labels)
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        sample_index = np.arange(0, n_temporal, 4)
        temporal_labels = temporal_labels[sample_index]
        features = (features / 4)[sample_index,:]
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        # Remove clients with too many missing values
        n_temporal, n_spatial = features.shape
        index = []
        threshold = 5000
        for i in range(n_spatial):
#            print(i, np.sum(features[:,i] == 0))
            if np.sum(features[:,i] == 0) > threshold:
                index.append(i)
        spatial_labels = np.delete(spatial_labels, index, -1)
        features = np.delete(features, index, -1)
        n_temporal, n_spatial = features.shape
        if args.debug:
            print(spatial_labels.shape)
            print(spatial_labels)
            print(temporal_labels.shape)
            print(temporal_labels)
            print(features.shape)
            print(features)
        df = pd.DataFrame({
            "date": np.repeat(temporal_labels, n_spatial), 
            "client": np.tile(spatial_labels, n_temporal), 
            "power_kWh": np.reshape(features, -1), 
        })
        if self.debug:
            print(df.loc[df["date"] == temporal_labels[0]])
            print(df.loc[df["date"] == temporal_labels[1]])
            print(df.loc[df["date"] == temporal_labels[-2]])
            print(df.loc[df["date"] == temporal_labels[-1]])
        pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
        pd.DataFrame({"client": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
        df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")

    def _convert(self, args):
        # Convert Electricity
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        if not all(paths_exist):
            #   Load original data
            path = self.spatiotemporal_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = pd.read_csv(path, header=None)
            if self.debug:
                print(df)
            #   Convert temporal labels
            temporal_labels = util.generate_temporal_labels(
                dt.datetime(2012, 1, 1),
                dt.datetime(2015, 1, 1),
                dt.timedelta(hours=1),
                bound_inclusion=[True, False]
            )
            temporal_labels = np.array(temporal_labels)
            n_temporal = temporal_labels.shape[0]
            #   Convert spatial labels
            spatial_labels = np.array([str(i+1) for i in range(len(df.columns))])
            n_spatial = spatial_labels.shape[0]
            #   Convert spatiotemporal features
            spatmp = df.to_numpy()
            if self.debug:
                print("Temporal Labels =", temporal_labels.shape)
                print(temporal_labels)
                print("Spatial Labels =", spatial_labels.shape)
                print(spatial_labels)
                print("Spatiotemporal =", spatmp.shape)
                print(spatmp)
            df = pd.DataFrame({
                "date": np.repeat(temporal_labels, n_spatial), 
                "client": np.tile(spatial_labels, n_temporal), 
                "power_kWh": np.reshape(spatmp, -1), 
            })
            if self.debug:
                print(df.loc[df["date"] == temporal_labels[0]])
                print(df.loc[df["date"] == temporal_labels[1]])
                print(df.loc[df["date"] == temporal_labels[-2]])
                print(df.loc[df["date"] == temporal_labels[-1]])
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            pd.DataFrame({"client": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")


class Solar_Energy(Integrator):

    def name(self):
        return "Solar-Energy"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "solar_AL.txt.gz"])

    def acquire(self, args):
        if not os.path.exists(self.spatiotemporal_features_inpath()):
            print("Download \"%s\" from sub-directory \"%s\" of repository at \"%s\" and save to acquisition directory \"%s\"" % (
                os.path.basename(self.spatiotemporal_features_inpath()), 
                "./solar-energy", 
                "https://github.com/laiguokun/multivariate-time-series-data", 
                self.acquire_dir(), 
            ))

    def convert(self, args):
        # Convert Solar-Energy
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        if not all(paths_exist):
            #   Load original data
            path = self.spatiotemporal_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = pd.read_csv(path, header=None)
            if self.debug:
                print(df)
            #   Convert temporal labels
            temporal_labels = util.generate_temporal_labels(
                dt.datetime(2006, 1, 1),
                dt.datetime(2007, 1, 1),
                dt.timedelta(minutes=10),
                bound_inclusion=[True, False]
            )
            temporal_labels = np.array(temporal_labels)
            n_temporal = temporal_labels.shape[0]
            #   Convert spatial labels
            spatial_labels = np.array([str(i+1) for i in range(len(df.columns))])
            n_spatial = spatial_labels.shape[0]
            #   Convert spatiotemporal features
            spatmp = df.to_numpy()
            if self.debug:
                print("Temporal Labels =", temporal_labels.shape)
                print(temporal_labels)
                print("Spatial Labels =", spatial_labels.shape)
                print(spatial_labels)
                print("Spatiotemporal =", spatmp.shape)
                print(spatmp)
            df = pd.DataFrame({
                "date": np.repeat(temporal_labels, n_spatial), 
                "plant": np.tile(spatial_labels, n_temporal), 
                "power_MW": np.reshape(spatmp, -1), 
            })
            if self.debug:
                print(df.loc[df["date"] == temporal_labels[0]])
                print(df.loc[df["date"] == temporal_labels[1]])
                print(df.loc[df["date"] == temporal_labels[-2]])
                print(df.loc[df["date"] == temporal_labels[-1]])
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            pd.DataFrame({"plant": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")


class METR_LA(Integrator):

    def name(self):
        return "METR-LA"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "metr-la.h5"])

    def spatial_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "graph_sensor_locations.csv"])

    def graph_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "adj_mx.pkl"])

    def hdf5_key(self):
        return "df"

    def start_date(self):
        return "2012-03-01_00-00-00"

    def acquire(self, args):
        paths_exist = [
            os.path.exists(self.spatiotemporal_features_inpath()), 
            os.path.exists(self.spatial_features_inpath()), 
            os.path.exists(self.graph_features_inpath()), 
        ]
        if not all(paths_exist):
            print(
                "Download \"%s\" from \"%s\" and save to acquisition directory \"%s\"" % (
                    os.path.basename(self.spatiotemporal_features_inpath()), 
                    "https://drive.google.com/drive/folders/10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX", 
                    self.acquire_dir(), 
                )
            )
            print(
                "Download \"%s\" and \"%s\" from sub-directory \"%s\" of repository at \"%s\" and save to acquisition directory \"%s\"" % (
                    os.path.basename(self.spatial_features_inpath()), 
                    os.path.basename(self.graph_features_inpath()), 
                    "./data/sensor_graph/", 
                    "https://github.com/liyaguang/DCRNN", 
                    self.acquire_dir(), 
                )
            )

    def convert(self, args):
        paths_exist = [
            os.path.exists(self.temporal_labels_outpath()), 
            os.path.exists(self.spatial_labels_outpath()), 
            os.path.exists(self.spatiotemporal_features_outpath()), 
        ]
        # Convert spatiotemporal features and labels
        if not all(paths_exist):
            f = h5py.File(self.spatiotemporal_features_inpath())
            spatial_labels = np.array(
                [
                    (str(label) if isinstance(label, np.int64) else label.decode("utf-8")) for label in f[self.hdf5_key()]["axis0"]
                ]
            )
            print(spatial_labels)
            sort_index = np.argsort(spatial_labels)
            print(sort_index)
            print(spatial_labels[sort_index])
            spatial_labels = spatial_labels[sort_index]
            n_spatial = spatial_labels.shape[0]
            features = np.array(list(f[self.hdf5_key()]["block0_values"]))[:,sort_index]
            n_temporal, n_spatial = features.shape
            temporal_labels = np.array(
                util.generate_temporal_labels(self.start_date(), n_temporal, [5, "minutes"])
            )
            data = {
                "date": np.repeat(temporal_labels, n_spatial), 
                "sensor": np.tile(spatial_labels, n_temporal), 
                "speedmph": np.reshape(features, -1)
            }
            df = pd.DataFrame(data)
            if args.debug:
                for i in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
                    print(temporal_labels[i], spatial_labels[0], features[i,0])
                print(df.iloc[np.arange(0, len(df), n_spatial)])
            pd.DataFrame({"sensor": spatial_labels}).to_csv(self.spatial_labels_outpath(), index=False)
            pd.DataFrame({"date": temporal_labels}).to_csv(self.temporal_labels_outpath(), index=False)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")
        #   Convert spatial features
        if not os.path.exists(self.spatial_features_outpath()):
            spatial_features_fname = os.path.basename(self.spatial_features_inpath())
            path = self.spatial_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            df = self.load_spatial_df(path)
            spatial_labels = pd.read_csv(self.spatial_labels_outpath()).to_numpy(dtype=str).reshape(-1)
            _spatial_labels = df["sensor"].to_numpy(dtype=str)
            print(spatial_labels)
            print(_spatial_labels)
            dat_sel = DataSelection()
            indices = dat_sel.indices_from_selection(_spatial_labels, ["literal"] + list(spatial_labels))
            df = df.iloc[indices,:]
            if self.debug:
                print(df)
            df.to_csv(self.spatial_features_outpath(), index=False)
        #   Convert graph topology
        if not os.path.exists(self.graph_features_outpath()):
            graph_features_fname = os.path.basename(self.graph_features_inpath())
            path = self.graph_features_inpath()
            assert os.path.exists(path), "Missing data file \"%s\"" % (path)
            in_path, out_path = path, os.sep.join([
                self.cache_dir(), 
                graph_features_fname.replace(".pkl", "_fixed.pkl")
            ])
            self.fix_file_content(in_path, out_path)
            path = out_path
            data = util.from_cache(path, encoding="bytes")
            spatial_labels = pd.read_csv(self.spatial_labels_outpath()).to_numpy(dtype=str).reshape(-1)
            label_index_map = util.to_dict([lab.decode("utf-8") for lab in data[1].keys()], data[1].values())
            adj = data[2]
            indices = np.array(util.get_dict_values(label_index_map, spatial_labels))
            if self.debug:
                print("LABEL INDEX MAP")
                print(label_index_map)
                print("SPATIAL LABELS")
                print(spatial_labels)
                print("SPATIAL LABEL INDICES")
                print(indices)
                print("ADJ")
                print(adj)
            dat_sel = DataSelection()
            adj = dat_sel.filter_axis(adj, [0, 1], [indices, indices])
            edgelist = W_to_edgelist(adj, spatial_labels)
            df = pd.DataFrame(
                {
                    "source": [edge[0] for edge in edgelist], 
                    "destination": [edge[1] for edge in edgelist], 
                    "weight": [edge[2] for edge in edgelist], 
                }
            )
            if self.debug:
                print(df)
            df.to_csv(self.graph_features_outpath(), index=False)

    def fix_file_content(self, in_path, out_path):
        content = ""
        with open(in_path, "rb") as in_file:
            content = in_file.read()
        out_size = 0
        with open(out_path, "wb") as out_file:
            for line in content.splitlines():
                out_size += len(line) + 1
                out_file.write(line + str.encode("\n"))

    def load_spatial_df(self, path):
        df = pd.read_csv(path)
        df = df.drop("index", axis=1)
        df = df.rename({"sensor_id": "sensor"}, axis=1)
        return df


class PEMS_BAY(METR_LA):

    def name(self):
        return "PEMS-BAY"

    def spatiotemporal_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "pems-bay.h5"])

    def spatial_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "graph_sensor_locations_bay.csv"])

    def graph_features_inpath(self):
        return os.sep.join([self.acquire_dir(), "adj_mx_bay.pkl"])

    def hdf5_key(self):
        return "speed"

    def start_date(self):
        return "2017-01-01_00-00-00"

    def load_spatial_df(self, path):
        return pd.read_csv(path, names=["sensor", "latitude", "longitude"])


class NEW_METR_LA(Integrator):

    def name(self):
        return "NEW-METR-LA"

    def download_pems(self, pems, district, ftype, years_and_months, download_dir):
        ftype_fname_map = {
            "station_5min": "d%02d_text_%s_%d_%02d_%02d.txt.gz", 
            "meta": "d%02d_text_%s_%d_%02d_%02d.txt", 
        }
        fname_template = ftype_fname_map[ftype]
        # Start
        for year, month in years_and_months:
            fnames = []
            for day in range(1, util.days_in_month(year, month)+1):
                fnames.append(fname_template % (district, ftype, year, month, day))
            paths = [os.sep.join([download_dir, _]) for _ in fnames]
            if not all([os.path.exists(_) for _ in paths]):
                ftd = pems.get_files(
                    start_year=year, end_year=year, districts=["%d" % (district)], 
                    file_types=[ftype], months=[util.month_name_map[month]], 
                )
                for i in range(len(ftd)):
                    path = os.sep.join([download_dir, ftd[i]["file_name"]])
                    if not os.path.exists(path):
                        pems._download_file(
                            file_name=ftd[i]["file_name"], file_url=ftd[i]["download_url"], 
                            save_path=download_dir, 
                        )

    def integrator(self):
        return METR_LA()

    def district(self):
        return 7

    def years_and_months(self):
        return ((2012, 3), (2012, 4), (2012, 5), (2012, 6))

    def var(self):
        return Container().set(
            [
                "replace_existing", 
                "file_range", 
                "debug", 
            ], 
            [
                True, 
                [0, -4], 
                0, 
            ], 
            multi_value=True
        )

    def acquire(self, args):
        if not ("username" in args or "password" in args):
            print("Please provde your Caltrans PeMS username and password (--username ... --password ...) to begin downloading.")
            return
        pems = PeMSHandler(username=args.username, password=args.password)
        self.download_pems(
            pems, 
            self.district(), 
            "station_5min", 
            self.years_and_months(), 
            self.acquire_dir(), 
        )
        try:
            self.download_pems(
                pems, 
                self.district(), 
                "meta", 
                self.years_and_months(), 
                self.acquire_dir(), 
            )
        except:
            pass

    def convert(self, args):
        var = self.var().merge(args)
        # Start
        pd.options.mode.chained_assignment = None
        warnings.simplefilter(action='ignore', category=FutureWarning)
        if 0:
            df_i = pd.read_csv(self.integrator().spatiotemporal_features_outpath())
            if var.debug:
                print(df_i)
            if var.debug:
                missing = get_missing(df_i, "sensor")
                print(missing)
            spatial_labels = df_i["sensor"].astype(str).unique()
            temporal_labels = df_i["date"].astype(str).unique()
            print(spatial_labels, len(spatial_labels))
            print(temporal_labels, len(temporal_labels))
            quit()
        if not os.path.exists(self.spatiotemporal_features_outpath()):
            df_i = pd.read_csv(self.integrator().spatiotemporal_features_outpath())
            df_i["sensor"] = df_i["sensor"].astype(str)
            df_i["date"] = df_i["date"].astype(str)
            if var.debug:
                print(df_i)
            if var.debug:
                missing = get_missing(df_i, "sensor")
                print(missing)
            spatial_labels = df_i["sensor"].astype(str).unique()
            temporal_labels = df_i["date"].astype(str).unique()
            n_spatial = len(spatial_labels)
            n_cols = 52
            cols = [
                "Timestamp", 
                "Station", 
                "District", 
                "Freeway", 
                "Direction", 
                "Lane_Type", 
                "Station_Length", 
                "Samples", 
                "Percent_Observed", 
                "Total_Flow", 
                "Avg_Occupancy", 
                "Avg_Speed", 
            ]
            for i in range(1, (n_cols - len(cols)) // 5 + 1):
                cols += [
                    "Lane_%d_Samples" % (i), 
                    "Lane_%d_Flow" % (i), 
                    "Lane_%d_Avg_Occupancy" % (i), 
                    "Lane_%d_Avg_Speed" % (i), 
                    "Lane_%d_Observed" % (i), 
                ]
            kept_cols = [
                "Timestamp", 
                "Station", 
                "Samples", 
                "Percent_Observed", 
                "Total_Flow", 
                "Avg_Occupancy", 
                "Avg_Speed", 
            ]
            # Start joining df_i and df_j into df_k
            n_temporal = 288
            paths = sorted(glob.glob(os.sep.join([self.acquire_dir(), "*station_5min*"])))
            if var.file_range[0] < 0:
                var.file_range[0] = len(paths) + var.file_range[0] + 1
            if var.file_range[1] < 0:
                var.file_range[1] = len(paths) + var.file_range[1] + 1
            paths = paths[var.file_range[0]:var.file_range[1]]
            orig_dfs = []
            for i, path in zip(range(var.file_range[0], var.file_range[1]), paths):
                print("%d/%d: %s" % (i-var.file_range[0]+1, len(paths), os.path.basename(path)))
                try:
                    orig_dfs.append(pd.read_csv(path, names=cols, usecols=kept_cols))
                except:
                    input("F")
            input("Ready")
            dfs = []
            for i, path in zip(range(var.file_range[0], var.file_range[1]), paths):
                print("%d/%d: %s" % (i-var.file_range[0]+1, len(paths), os.path.basename(path)))
                start, end = n_temporal * n_spatial * i, n_temporal * n_spatial * (i+1)
#                df_j = pd.read_csv(path, names=cols, usecols=kept_cols)
                df_j = orig_dfs[i]
                df_j["Timestamp"] = pd.to_datetime(df_j["Timestamp"]).dt.strftime("%Y-%m-%d_%H-%M-%S")
                df_j["Station"] = df_j["Station"].astype(str) 
                if var.debug:
                    print("df_j =")
                    print(df_j)
                df_k = pd.DataFrame(columns=df_j.columns, index=range(n_temporal*n_spatial))
                df_k[["Timestamp", "Station", "Avg_Speed"]] = df_i.loc[start:end-1,["date", "sensor", "speedmph"]].reset_index(drop=True)
                if var.debug:
                    print(df_k)
                _temporal_labels = df_j["Timestamp"].unique()
                _n_temporal = len(_temporal_labels)
                if True or _n_temporal != n_temporal:
                    if var.debug:
                        print("unexpected time-steps in with n_temporal=%s" % (_n_temporal))
                    for idx, temporal_label in enumerate(temporal_labels[n_temporal*i:n_temporal*(i+1)]):
                        if temporal_label in _temporal_labels:
                            tmp_j = df_j.loc[df_j["Timestamp"] == temporal_label]
                            if var.debug:
                                print("tmp_j =")
                                print(tmp_j)
                            start, end = n_spatial * idx, n_spatial * (idx + 1)
                            tmp_k = df_k.iloc[start:end]
                            if var.debug:
                                print("tmp_k =")
                                print(tmp_k)
                            tmp_k.loc[tmp_k["Station"].isin(tmp_j["Station"])] = tmp_j.loc[tmp_j["Station"].isin(tmp_k["Station"])].set_index(
                                tmp_k.loc[tmp_k["Station"].isin(tmp_j["Station"])].index
                            )
                            if var.debug:
                                print("tmp_k =")
                                print(tmp_k)
                            df_k.iloc[start:end] = tmp_k
                            if var.debug:
                                print("tmp_k =")
                                print(df_k.iloc[start:end])
                                input()
                else:
                    mask = df_k["Station"].isin(df_j["Station"])
                    df_k.loc[mask] = df_j.loc[df_j["Station"].isin(spatial_labels)].set_index(mask[mask].index)
                if not var.replace_existing:
                    df_k[["Avg_Speed"]] = df_i.loc[start:end-1,["speedmph"]].reset_index(drop=True)
                if var.debug:
                    print("df_k =")
                    print(df_k)
                if var.debug:
                    missing = get_missing(df_k, "Station")
                    print("missing =")
                    print(missing)
                dfs.append(df_k)
                if var.debug:
                    input()
            df = pd.concat(dfs)
            df.dropna(subset=["Timestamp"], inplace=True)
            df.to_csv(self.spatiotemporal_features_outpath(), index=False, compression="gzip")
        if not os.path.exists(self.spatial_features_outpath()):
            df_i = pd.read_csv(self.integrator().spatial_features_outpath())
            cols = ["ID", "Fwy", "Dir", "District", "County", "City", "State_PM", "Abs_PM", "Latitude", "Longitude", "Length", "Type", "Lanes", "Name", "User_ID_1", "User_ID_2", "User_ID_3", "User_ID_4"]
            paths = sorted(glob.glob(os.sep.join([self.acquire_dir(), "*meta*"])))
            df_j = pd.read_csv(paths[-1], usecols=cols[:-4], sep="\t")
            df_j.columns = ["Station", "Freeway", "Direction"] + list(df_j.columns)[3:]
            if var.debug:
                print(df_j)
            df_k = pd.DataFrame(columns=df_j.columns, index=df_i.index)
            df_k[["Station", "Latitude", "Longitude"]] = df_i[["sensor", "latitude", "longitude"]]
            if var.debug:
                print(df_k)
            df_k.loc[df_k["Station"].isin(df_j["Station"])] = df_j.loc[df_j["Station"].isin(df_k["Station"])].set_index(
                df_k.loc[df_k["Station"].isin(df_j["Station"])].index
            )
            if not var.replace_existing:
                df_k[["Latitude", "Longitude"]] = df_i[["latitude", "longitude"]]
            if var.debug:
                print(df_k)
            if var.debug:
                print(get_missing(df_k, "Station"))
            df_k.to_csv(self.spatial_features_outpath(), index=False)
        df = pd.read_csv(self.integrator().spatial_labels_outpath())
        df.columns = ["Station"]
        df.to_csv(self.spatial_labels_outpath(), index=False)
        df = pd.read_csv(self.integrator().temporal_labels_outpath())
        df.columns = ["Timestamp"]
        df.to_csv(self.temporal_labels_outpath(), index=False)
        if os.path.exists(self.integrator().graph_features_outpath()) and not os.path.exists(self.graph_features_outpath()):
            shutil.copy(self.integrator().graph_features_outpath(), self.graph_features_outpath())



class NEW_PEMS_BAY(NEW_METR_LA):

    def name(self):
        return "NEW-PEMS-BAY"

    def integrator(self):
        return PEMS_BAY()

    def district(self):
        return 4

    def years_and_months(self):
        return ((2017, 1), (2017, 2), (2017, 3), (2017, 4), (2017, 5), (2017, 6))

    def var(self):
        return Container().set(
            [
                "replace_existing", 
                "file_range", 
                "debug", 
            ], 
            [
                True, 
                [0, -1], 
                0, 
            ], 
            multi_value=True
        )
