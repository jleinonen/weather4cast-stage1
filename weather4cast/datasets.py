from datetime import datetime, timedelta
import os

import h5py
import netCDF4
import numpy as np
from tensorflow.keras.utils import Sequence


file_dir = os.path.dirname(os.path.abspath(__file__))


def sequence_start_times(data_dir, dataset="CTTH",
    sequence_length=4+32, interval=timedelta(minutes=15)):
    """Find all times at which a valid sequence of 4+32 frames is available.
    """

    if data_dir is None:
        data_dir = os.path.join(
            file_dir,
            "../data/w4c-core-stage-1/R1/training/"
        )

    def time_from_file(fn):
        timestamp = fn.split(".")[0].split("_")[-1]
        return datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ")

    # build index of available time steps
    time_index = set()
    for subdir in os.listdir(data_dir):
        day_dir = os.path.join(data_dir, subdir, dataset)
        for fn in os.listdir(day_dir):
            time = time_from_file(fn)
            time_index.add(time)

    # filter out times that are not a valid starting time for a sequence
    def starts_valid_sequence(time):
        t = time
        for i in range(sequence_length-1):
            t += interval
            if t not in time_index:
                return False
        return True

    return sorted(t for t in time_index if starts_valid_sequence(t))


class DataReader:
    def __init__(self, fill_value=0.0):
        self.fill_value = fill_value
        self.cache = {}

    def read_file(self, fn):
        with open(fn, mode='rb') as f:
            return f.read()

    def preprocess(self, data, var_name, valid_range, data_fillvalue):

        fill = (data == data_fillvalue)
        data = data.astype(dtype=np.float32)
        data -= valid_range[0]
        data *= 1./(valid_range[1]-valid_range[0])
        if var_name == "temperature":
            if not fill.all():
                data[fill] = data[~fill].mean()
            else:
                data[fill] = self.fill_value
        else:
            data[fill] = self.fill_value
        
        return data

    def get_data(self, fn, variables, box):
        ((i0,i1),(j0,j1)) = box       
        
        ds = None
        var_data = []

        try:
            for var_name in variables:
                if (fn, var_name) not in self.cache:
                    if ds is None:
                        try:
                            raw_data = self.read_file(fn)
                        except FileNotFoundError:
                            fn_mod = fn.replace("MSG4", "MSG2")
                            raw_data = self.read_file(fn_mod)
                        
                        ds = netCDF4.Dataset(None, 'r', memory=raw_data)
                        ds.set_auto_scale(False)
                    var = ds[var_name]
                    data = np.array(var[:])
                    valid_range = var.valid_range
                    fill_value = var._FillValue
                    data = self.preprocess(data, var_name, valid_range, fill_value)                    
                    self.cache[(fn, var_name)] = (data, valid_range, fill_value)
                else:
                    (data, valid_range, fill_value) = self.cache[(fn, var_name)]
                
                data = data[i0:i1,j0:j1]
                
                var_data.append(data)

        finally:
            if ds is not None:
                ds.close()

        return np.stack(var_data, axis=-1)


region_coordinates = {
    "R3": (935,400),
    "R6": (1270,250),
    "R2": (1550,200),
    "R1": (1850,760),
    "R5": (1300,550),
    "R4": (1020,670)
}

region_size = (256,256)

class StaticData:
    def __init__(self, data_dir=None, regions=None):
        if data_dir is None:
            data_dir = os.path.join(
                file_dir,
                "../data/static/"
            )
        
        fn_latlon = os.path.join(data_dir, 
            "Navigation_of_S_NWC_CT_MSG4_Europe-VISIR_20201106T120000Z.nc")
        with netCDF4.Dataset(fn_latlon, 'r') as ds:
            self.lon = np.array(ds["longitude"][0,:,:]).astype(np.float32)
            self.lat = np.array(ds["latitude"][0,:,:]).astype(np.float32)
        self.lon = (self.lon+76) / (76+76)
        self.lat = (self.lat-23) / (86-23)

        fn_elev = os.path.join(data_dir, 
            "S_NWC_TOPO_MSG4_+000.0_Europe-VISIR.raw")
        self.elevation = np.fromfile(fn_elev, dtype=np.float32).reshape(self.lon.shape)
        self.elevation[self.elevation < 0] = 0
        self.elevation /= self.elevation.max()

    def get_data(self, variables, region, box):
        (reg_j0, reg_i0) = region_coordinates[region]
        reg_i1 = reg_i0 + region_size[0]
        reg_j1 = reg_j0 + region_size[1]
        ((i0,i1),(j0,j1)) = box
        var = {
            "latitude": self.lat,
            "longitude": self.lon,
            "elevation": self.elevation
        }
        data = [var[v][reg_i0:reg_i1,reg_j0:reg_j1][i0:i1,j0:j1] for v in variables]
        return np.stack(data, axis=-1)


class BatchGenerator(Sequence):
    def __init__(self, 
        predictors=None,
        targets=None,
        data_dir=None,
        comp_dir="w4c-core-stage-1",
        regions=None,
        sequence_length=(4,32),
        box_shape=(256,256),        
        batch_size=32,
        interval=timedelta(minutes=15),
        orig_box_shape=(256,256),
        data_subset="training",
        random_seed=None,
        augment=True,
        shuffle=True
    ):
        default_predictors = {
            "CTTH": ["temperature"],
            "CRR": ["crr_intensity"],
            "ASII": ["asii_turb_trop_prob"], 
            "CMA": ["cma"],
            "static": ["elevation", "longitude", "latitude"]
        }
        if predictors is None:
            predictors = default_predictors
        default_targets = {
            "CTTH": ["temperature"],
            "CRR": ["crr_intensity"],
            "ASII": ["asii_turb_trop_prob"], 
            "CMA": ["cma"]
        }    
        if targets is None:
            targets = default_targets
        self.variables = (predictors, targets)

        self.static_data = StaticData(data_dir=data_dir)
        if data_dir is None:
            data_dir = os.path.join(
                file_dir,
                "../data/",
                comp_dir
            )
        self.data_dir = data_dir
        if regions is None:
            regions = ["R1", "R2", "R3"]
        self.regions = regions
        self.data_subset = data_subset
        self.data_reader = DataReader()        

        self.sequence_length = sequence_length
        self.box_shape = box_shape
        self.batch_size = batch_size
        self.num_vars = tuple(
            sum(len(v[k]) for k in v)
            for v in self.variables
        )
        self.batch_shape = tuple(
            (batch_size, s) + box_shape + (n,) 
            for (s,n) in zip(self.sequence_length, self.num_vars)
        )
        self.interval = interval
        self.orig_box_shape = orig_box_shape
        
        self.start_times = []
        for dataset in (set(predictors) & set(targets)):
            for region in self.regions:
                self.start_times.append(sequence_start_times(
                    data_dir=os.path.join(data_dir, region, data_subset),
                    sequence_length=sum(sequence_length),
                    interval=interval,
                    dataset=dataset
                ))
        self.start_times = sorted(
            set.intersection(*(set(st) for st in self.start_times))
        )
        self.n_times = len(self.start_times)
        self.n = self.n_times * len(self.regions)
        self.shuffle = shuffle
        self.augment_batch = augment
        self.rng = np.random.RandomState(seed=random_seed)
        self.on_epoch_end()

    def __len__(self):
        return self.n // self.batch_size

    def on_epoch_end(self):
        self.indices = np.arange(self.n)
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def _file_path(self, dataset, region, time):
        file_ds = "ASII-TF" if dataset == "ASII" else dataset
        timestamp = time.strftime("%Y%m%dT%H%M%SZ")
        datestamp = time.strftime("%Y%j")
        fn = "S_NWC_{}_MSG4_Europe-VISIR_{}.nc".format(file_ds, timestamp)
        path = os.path.join(
            self.data_dir, # e.g. w4c-core-stage-1
            region, # e.g. R1
            self.data_subset, # e.g. training,
            datestamp, # e.g. 2019204,
            dataset, # e.g. CTTH
            fn # e.g. S_NWC_CTTH_MSG4_Europe-VISIR_20190723T060000Z.nc
        )
        return path

    def _random_box(self):        
        i0 = self.rng.randint(self.orig_box_shape[0]-self.box_shape[0]+1)
        i1 = i0 + self.box_shape[0]
        j0 = self.rng.randint(self.orig_box_shape[1]-self.box_shape[1]+1)
        j1 = j0 + self.box_shape[1]
        return ((i0,i1), (j0,j1))

    def _random_augments(self):
        if self.augment_batch:
            rotate = self.rng.randint(4)
            flipud = bool(self.rng.randint(2))
            fliplr = bool(self.rng.randint(2))
        else:
            rotate = 0
            flipud = False
            fliplr = False
        return (rotate, flipud, fliplr)

    def augment(self, batch, augments):
        (rotate, flipud, fliplr) = augments
        if rotate > 0:
            batch = np.rot90(batch, k=rotate, axes=(2,3))
        if flipud:
            batch = batch[:,:,::-1,:,:]
        if fliplr:
            batch = batch[:,:,:,::-1,:]
        return batch

    def var_at_time(self, dataset, variable, region, time, box=None):
        path = self._file_path(dataset, region, time)
        return self.data_reader.get_data(path, variable, box)

    def __getitem__(self, idx):
        """Build a training batch by reading data from the files.
        """
        k0 = idx * self.batch_size
        k1 = k0 + self.batch_size
        indices = self.indices[np.arange(k0,k1)]
        regions = indices // self.n_times
        indices_in_region = indices % self.n_times

        batches = [np.empty(s, dtype=np.float32) for s in self.batch_shape]
        box = self._random_box()
        augments = self._random_augments()

        for (num_batch, (batch, variables)) in enumerate(zip(batches, self.variables)):
            data_parts = []
            slices = []

            for (k_batch, (reg, k_reg)) in enumerate(zip(regions, indices_in_region)):
                time = self.start_times[k_reg]
                if num_batch == 1:
                    time += self.sequence_length[0] * self.interval
                reg_name = self.regions[reg]
                for t in range(batch.shape[1]):
                    v0 = 0
                    for dataset in variables:
                        if dataset == "static":
                            data = self.static_data.get_data(variables[dataset], reg_name, box)
                        else:
                            data = self.var_at_time(dataset, variables[dataset], reg_name, time, box)
                        v1 = v0 + len(variables[dataset])
                        nd_slice = (k_batch, t, slice(None), slice(None), slice(v0,v1))
                        data_parts.append(data)
                        slices.append(nd_slice)
                        v0 = v1
                    time += self.interval
            
            for (nd_slice, data) in zip(slices, data_parts):
                batch[nd_slice] = data
        
        batches[0] = self.augment(batches[0], augments)
        batches[1] = self.augment(batches[1], augments)

        # the model expects a list of outputs because losses may be different...
        batches[1] = [batches[1][...,i:i+1] for i in range(batches[1].shape[-1])]

        return tuple(batches)


def setup_univariate_batch_gen(batch_gen, dataset, var_name,
    batch_size=None):
    
    batch_gen.variables = (
        batch_gen.variables[0],
        {dataset: [var_name]}
    )
    batch_gen.num_vars = (batch_gen.num_vars[0], 1)
    batch_gen.batch_shape = (
        batch_gen.batch_shape[0],
        batch_gen.batch_shape[1][:-1] + (1,)
    )
    if batch_size is not None:
        batch_gen.batch_size = batch_size
        batch_gen.batch_shape = (
            (batch_size,) + batch_gen.batch_shape[0][1:],
            (batch_size,) + batch_gen.batch_shape[1][1:],
        )


postproc_scaling = {
    "temperature": ((0, 22000), 65535),
    "crr_intensity": ((0, 500), 65535),
    "asii_turb_trop_prob": ((0, 100), 255),
    "cma": ((0, 1), 255)
}


def postprocess(data, var_name):
    (valid_range, fill_value) = postproc_scaling[var_name]
    #if var_name == "asii_turb_trop_prob":
    #    data = inv_normlogit(data)
    data *= (valid_range[1]-valid_range[0])
    data += valid_range[0]
    data = data.round().astype(np.uint16)
    return data


def generate_submission(model, location, 
    past_timesteps=4, interval=timedelta(minutes=15),
    regions=None, comp_dir=None
    ):

    box_shape = (256, 256)
    batch_gen = BatchGenerator(
        data_subset="test",
        sequence_length=(4,0),
        box_shape=box_shape,
        shuffle=False,
        batch_size=1,
        comp_dir=comp_dir,
        regions=regions
    )

    predictors = batch_gen.variables[0]
    num_variables = sum(len(predictors[ds]) for ds in predictors)
    target_vars = ["temperature", "crr_intensity", "asii_turb_trop_prob", "cma"]

    box = ((0,256),(0,256))
    for region in batch_gen.regions:
        for time0 in batch_gen.start_times:
            batch_shape = (1,past_timesteps) + box_shape + (num_variables,)
            batch = np.zeros(batch_shape, dtype=np.float32)
            for i in range(4):
                time = time0 + i*interval
                k = 0
                for dataset in predictors:
                    for variable in predictors[dataset]:
                        if dataset == "static":
                            data = batch_gen.static_data.get_data([variable], region, box)
                        else:
                            path = batch_gen._file_path(dataset, region, time)
                            if not (os.path.isfile(path) or os.path.isfile(path.replace("MSG4", "MSG2"))):
                                path = path.replace(
                                    "/"+time.strftime("%Y%j")+"/",
                                    "/"+(time-timedelta(days=1)).strftime("%Y%j")+"/",
                                )
                            data = batch_gen.data_reader.get_data(path, [variable], box)
                        batch[0,i,:,:,k:k+1] = data
                        k += 1

            predictions = model.predict(batch)
            data_out = []
            for (prediction, var_name) in zip(predictions, target_vars):
                prediction = postprocess(prediction, var_name)
                prediction = prediction[0,...]
                prediction = prediction.transpose(0,3,1,2)
                data_out.append(prediction)
            data_out = np.concatenate(data_out, axis=1)

            fdir = os.path.join(location, region, "test")
            if (time0.hour == 0) and (time0.minute==0):
                # some weirdness in the test set time stamps
                timestamp = (time0-timedelta(days=1)).strftime("%Y%j")
            else:
                timestamp = time0.strftime("%Y%j")
            fn = os.path.join(fdir, timestamp+".h5")
            os.makedirs(fdir, exist_ok=True)
            with h5py.File(fn, 'w', libver='latest') as f:
                f.create_dataset('array', shape=(data_out.shape), data=data_out, 
                    dtype=np.uint16, compression='gzip', compression_opts=9)
