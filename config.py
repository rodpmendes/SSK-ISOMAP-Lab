import sklearn.datasets as skdata
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Any
import numpy as np
from MedMNIST.WrappedMedMNIST import WrappedMedMNIST

class NoiseType(Enum):
    """
    Enumeration of noise types applicable to datasets.
    """
    NONE = "none"
    GAUSSIAN = "gaussian"
    SALT_PEPPER = "salt_pepper"
    POISSON = "poisson"
    SPECKLE = "speckle"
    UNIFORM = "uniform"
    SWAP = "swap"
    MASKING = "masking"

class ClusterType(Enum):
    """
    Enumeration of clustering algorithms.
    """
    GMM = "gmm"
    KMEANS = "kmeans"
    AGGLOMERATIVE = "agglomerative"

@dataclass
class DatasetConfig:
    """
    Configuration for loading and preprocessing a dataset.

    Data is loaded lazily upon accessing `.data`.

    Attributes:
        name (str): Dataset name or identifier.
        data_source (dict): Data source parameters.
        reduce_samples (bool): Whether to reduce the number of samples. Defaults to False.
        sample_percentage (float): Fraction of samples to keep if reduction is applied. Defaults to 0.0.
        reduce_dim (bool): Whether to reduce feature dimensionality. Defaults to False.
        num_features (int): Target number of features after dimensionality reduction. Defaults to 0.
    """
    
    name: str
    data_source: dict
    reduce_samples: bool = False
    sample_percentage: float = 0.0
    reduce_dim: bool = False
    num_features: int = 0

    _data: Any = field(default=None, init=False, repr=False)

    @property
    def data(self) -> Any:
        """
        Loads and returns the dataset, caching it after first load.

        Returns:
            Any: The loaded dataset object.

        Raises:
            ValueError: If loader function in data_source is invalid.
            RuntimeError: If fetching dataset from OpenML fails.
        """
        if self._data is None:
            if isinstance(self.data_source, WrappedMedMNIST):
                self._data = self.data_source.data  # Usa o atributo data de WrappedMedMNIST
            elif self.data_source.get("builtin", False):
                loader = self.data_source.get("loader")
                if not callable(loader):
                    raise ValueError(
                        f"Loader function not found or invalid in data_source for dataset '{self.name}'."
                    )
                self._data = loader()
            else:
                try:
                    self._data = skdata.fetch_openml(**self.data_source)
                except Exception as e:
                    raise RuntimeError(f"Error fetching OpenML dataset '{self.name}': {e}")
        return self._data
    

@dataclass
class NoiseConfig:
    """
    Configuration parameters for noise addition.

    Attributes:
        apply_noise (bool): Flag to indicate whether noise should be applied. Defaults to True.
        noise_type (NoiseType): Type of noise to add to the dataset. Defaults to NoiseType.NONE.
        max_std_dev (float): Maximum standard deviation for noise magnitude. Defaults to 1.0.
    """
    apply_noise: bool = True
    noise_type: NoiseType = NoiseType.NONE
    max_std_dev: float = 1.0
    
    def get_magnitudes(self, num_levels: int = 11) -> np.ndarray:
        """Generate noise magnitudes based on current configuration.

        Args:
            num_levels (int): Number of noise levels to generate. Default is 11.

        Returns:
            np.ndarray: Array of noise magnitudes.
        """
        if self.apply_noise and self.noise_type != NoiseType.NONE:
            return np.linspace(0, self.max_std_dev, num_levels)
        return np.array([0.0])
    
    def should_apply_noise(self) -> bool:
        """Check if noise should be applied based on configuration.

        Returns:
            bool: True if noise should be applied, False otherwise.
        """
        return self.apply_noise and self.noise_type != NoiseType.NONE


@dataclass
class ClusteringConfig:
    """
    Configuration of clustering algorithm options.

    Attributes:
        cluster_types (List[ClusterType]): List of allowed clustering methods.
        chosen_cluster (ClusterType): Currently selected clustering method.
    """
    cluster_types: List[ClusterType] = field(default_factory=lambda: [
        ClusterType.GMM,
        ClusterType.KMEANS,
        ClusterType.AGGLOMERATIVE])
    chosen_cluster: ClusterType = ClusterType.GMM

@dataclass
class ExperimentConfig:
    """
    Full experiment configuration containing datasets, noise, clustering, and other parameters.

    Attributes:
        datasets (List[DatasetConfig]): List of dataset configurations.
        noise_config (NoiseConfig): Noise-related configuration.
        clustering_config (ClusteringConfig): Clustering-related configuration.
        random_seed (int): Random seed for reproducibility. Defaults to 42.
        verbose (bool): Flag to enable verbose output/logging. Defaults to True.
        file_results (Optional[str]): Optional filename to save experiment results.
    """
    datasets: List[DatasetConfig] = field(default_factory=list)
    noise_config: NoiseConfig = field(default_factory=NoiseConfig)
    clustering_config: ClusteringConfig = field(default_factory=ClusteringConfig)
    random_seed: int = 42
    verbose: bool = True
    file_results: Optional[str] = None

    def validate(self) -> None:
        """
        Validate the experiment configuration.

        Raises:
            ValueError: If noise is enabled but noise type is NONE,
                        or chosen cluster is not in allowed cluster_types.
        """
        if self.noise_config.apply_noise and self.noise_config.noise_type == NoiseType.NONE:
            raise ValueError("Noise type cannot be NONE when noise is applied.")

        if self.clustering_config.chosen_cluster not in self.clustering_config.cluster_types:
            allowed = [c.value for c in self.clustering_config.cluster_types]
            raise ValueError(
                f"Chosen cluster '{self.clustering_config.chosen_cluster.value}' is not in allowed cluster types {allowed}"
            )
            
    def __post_init__(self) -> None:
        self.validate()
            

def openml(name: str, version: int = 1, as_frame: bool = True) -> dict:
    """
    Returns parameters to load an OpenML dataset.

    Args:
        name (str): Dataset name on OpenML.
        version (int, optional): Dataset version. Defaults to 1.
        as_frame (bool, optional): Load data as pandas DataFrame. Defaults to True.

    Returns:
        dict: Dictionary with OpenML loading parameters.
    """
    return {"name": name, "version": version, "as_frame": as_frame}

def builtin(loader: callable) -> dict:
    """
    Returns parameters for a built-in sklearn dataset loader.

    Args:
        loader (callable): Loader function that returns the dataset.

    Returns:
        dict: Dictionary specifying a builtin loader.
    """
    return {"loader": loader, "builtin": True}

def load_all_datasets() -> List[DatasetConfig]:
    """
    Creates a list of dataset configurations with lazy loading.

    Returns:
        List[DatasetConfig]: List of dataset configurations.
    """
    datasets = [
        DatasetConfig(name='digits', data_source=builtin(skdata.load_digits)),
        DatasetConfig(name='vowel', data_source=openml("vowel", version=2)),
        DatasetConfig(name='diabetes', data_source=openml("diabetes")),
        DatasetConfig(name='mfeat-karhunen', data_source=openml("mfeat-karhunen")),
        DatasetConfig(name='grub-damage', data_source=openml("grub-damage", version=2)),
        DatasetConfig(name='banknote-authentication', data_source=openml("banknote-authentication")),
        DatasetConfig(name='wall-robot-navigation', data_source=openml("wall-robot-navigation")),
        DatasetConfig(name='waveform-5000', data_source=openml("waveform-5000")),
        DatasetConfig(name='nursery', data_source=openml("nursery"), reduce_samples=True, sample_percentage=0.3),
        DatasetConfig(name='eye_movements', data_source=openml("eye_movements"), reduce_samples=True, sample_percentage=0.3),
        DatasetConfig(name='thyroid-dis', data_source=openml("thyroid-dis")),
        DatasetConfig(name='servo', data_source=openml("servo")),
        DatasetConfig(name='car-evaluation', data_source=openml("car-evaluation")),
        DatasetConfig(name='breast-tissue', data_source=openml("breast-tissue", version=2)),
        DatasetConfig(name='Engine1', data_source=openml("Engine1")),
        DatasetConfig(name='xd6', data_source=openml("xd6")),
        DatasetConfig(name='heart-h', data_source=openml("heart-h", version=3)),
        DatasetConfig(name='steel-plates-fault', data_source=openml("steel-plates-fault", version=3)),
        DatasetConfig(name='PhishingWebsites', data_source=openml("PhishingWebsites"), reduce_samples=True, sample_percentage=0.1),
        DatasetConfig(name='satimage', data_source=openml("satimage"), reduce_samples=True, sample_percentage=0.25),
        DatasetConfig(name='led24', data_source=openml("led24"), reduce_samples=True, sample_percentage=0.15),
        DatasetConfig(name='hayes-roth', data_source=openml("hayes-roth", version=2)),
        DatasetConfig(name='rabe_131', data_source=openml("rabe_131", version=2)),
        DatasetConfig(name='prnn_synth', data_source=openml("prnn_synth")),
        DatasetConfig(name='visualizing_environmental', data_source=openml("visualizing_environmental", version=2)),
        DatasetConfig(name='diggle_table_a2', data_source=openml("diggle_table_a2", version=2)),
        DatasetConfig(name='newton_hema', data_source=openml("newton_hema", version=2)),
        DatasetConfig(name='wisconsin', data_source=openml("wisconsin", version=2)),
        DatasetConfig(name='fri_c4_250_100', data_source=openml("fri_c4_250_100", version=2)),
        DatasetConfig(name='conference_attendance', data_source=openml("conference_attendance")),
        DatasetConfig(name='tic-tac-toe', data_source=openml("tic-tac-toe")),
        DatasetConfig(name='qsar-biodeg', data_source=openml("qsar-biodeg")),
        DatasetConfig(name='spambase', data_source=openml("spambase"), reduce_samples=True, sample_percentage=0.25),
        DatasetConfig(name='cmc', data_source=openml("cmc")),
        DatasetConfig(name='heart-statlog', data_source=openml("heart-statlog")),
        DatasetConfig(name='cnae-9', data_source=openml("cnae-9"), reduce_dim=True, num_features=50),
        DatasetConfig(name='AP_Breast_Kidney', data_source=openml("AP_Breast_Kidney"), reduce_dim=True, num_features=500),
        DatasetConfig(name='AP_Endometrium_Breast', data_source=openml("AP_Endometrium_Breast"), reduce_dim=True, num_features=400),
        DatasetConfig(name='AP_Ovary_Lung', data_source=openml("AP_Ovary_Lung"), reduce_dim=True, num_features=100),
        DatasetConfig(name='OVA_Uterus', data_source=openml("OVA_Uterus"), reduce_dim=True, num_features=100),
        DatasetConfig(name='micro-mass', data_source=openml("micro-mass"), reduce_dim=True, num_features=100),
        DatasetConfig(name='har', data_source=openml("har"), reduce_samples=True, sample_percentage=0.1, reduce_dim=True, num_features=100),
        DatasetConfig(name='eating', data_source=openml("eating"), reduce_dim=True, num_features=100),
        DatasetConfig(name='oh5.wc', data_source=openml("oh5.wc",version=1, as_frame=False), reduce_dim=True, num_features=40),
        DatasetConfig(name='leukemia', data_source=openml("leukemia"), reduce_dim=True, num_features=40),
        DatasetConfig(name='pendigits', data_source=openml("pendigits"), reduce_samples=True, sample_percentage=0.25),
        DatasetConfig(name='mnist_784', data_source=openml("mnist_784"), reduce_samples=True, sample_percentage=0.25, reduce_dim=True, num_features=20),
        DatasetConfig(name='Fashion-MNIST', data_source=openml("Fashion-MNIST"), reduce_samples=True, sample_percentage=0.5, reduce_dim=True, num_features=100),
        DatasetConfig(name='adult', data_source=openml('adult'), reduce_samples=True, sample_percentage=0.25, reduce_dim=True, num_features=100),
    ]
    return datasets

def load_one_dataset() -> List[DatasetConfig]:
    """
    Creates a list containing configuration for a single dataset using lazy loading.

    Returns:
        List[DatasetConfig]: List with one dataset configuration.
    """
    datasets = [
        DatasetConfig(name='iris', data_source=openml("iris"))
    ]
    return datasets


def load_tree_datasets() -> List[DatasetConfig]:
    """
    Creates a list containing configuration for tree dataset using lazy loading.

    Returns:
        List[DatasetConfig]: List with one dataset configuration.
    """
    datasets = [
        DatasetConfig(name='iris', data_source=openml("iris")),
        DatasetConfig(name='digits', data_source=builtin(skdata.load_digits)),
        DatasetConfig(name='wine-quality-red', data_source=openml('breast-cancer')),
    ]
    return datasets

def load_small_datasets() -> List[DatasetConfig]:
    """
    Creates a list containing configuration for small dataset using lazy loading.

    Returns:
        List[DatasetConfig]: List with one dataset configuration.
    """
    datasets = [
        DatasetConfig(name='iris', data_source=openml('iris')),
        DatasetConfig(name='wine', data_source=openml('wine'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/178),
        DatasetConfig(name='wine-quality-red', data_source=openml('wine-quality-red'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/1599),
        DatasetConfig(name='wine-quality-white', data_source=openml('wine-quality-white'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/4898),
        DatasetConfig(name='ecoli', data_source=openml('ecoli'), reduce_samples=True, sample_percentage=150 / 336),
        DatasetConfig(name='ionosphere', data_source=openml('ionosphere'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/351),
        DatasetConfig(name='seeds', data_source=openml('seeds'), reduce_samples=True, sample_percentage=150 / 210),
        DatasetConfig(name='glass', data_source=openml('glass'), reduce_samples=True, sample_percentage=150/214),
        DatasetConfig(name='pima-indians-diabetes', data_source=openml('diabetes'), reduce_samples=True, sample_percentage=150/768),
        DatasetConfig(name='balance-scale', data_source=openml('balance-scale'), reduce_samples=True, sample_percentage=150/625),
        DatasetConfig(name='vehicle', data_source=openml('vehicle'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/846),
        DatasetConfig(name='zoo', data_source=openml('zoo')),
        DatasetConfig(name='breast-w', data_source=openml('breast-w'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/699),
        DatasetConfig(name='spectf', data_source=openml('spectf'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/349),
        DatasetConfig(name='yeast', data_source=openml('yeast'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/1484),
        DatasetConfig(name='car', data_source=openml('car'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/1728),
        DatasetConfig(name='cleveland', data_source=openml('cleveland'), reduce_dim=True, num_features=10, reduce_samples=True, sample_percentage=150/303),
    ]
    return datasets

def load_small_datasets_not_reduced() -> List[DatasetConfig]:
    """
    Creates a list containing configuration for small not reduced dataset using lazy loading.

    Returns:
        List[DatasetConfig]: List with one dataset configuration.
    """
    datasets = [
        DatasetConfig(name='iris', data_source=openml('iris')),
        DatasetConfig(name='seeds', data_source=openml('seeds')),
        DatasetConfig(name='glass', data_source=openml('glass')),
        DatasetConfig(name='ecoli', data_source=openml('ecoli')),
        DatasetConfig(name='ionosphere', data_source=openml('ionosphere')),
        DatasetConfig(name='pima-indians-diabetes', data_source=openml('diabetes')),
        DatasetConfig(name='balance-scale', data_source=openml('balance-scale')),
        DatasetConfig(name='vehicle', data_source=openml('vehicle')),
        DatasetConfig(name='breast-w', data_source=openml('breast-w')),
        DatasetConfig(name='spectf', data_source=openml('spectf')),
        DatasetConfig(name='car', data_source=openml('car')),
        DatasetConfig(name='cleveland', data_source=openml('cleveland')),
        DatasetConfig(name='zoo', data_source=openml('zoo')),
        DatasetConfig(name='wine', data_source=openml('wine')),
        DatasetConfig(name='wine-quality-red', data_source=openml('wine-quality-red')),
        #DatasetConfig(name='wine-quality-white', data_source=openml('wine-quality-white')),
    ]
    return datasets

def load_paper_datasets() -> List[DatasetConfig]:
    """
    Creates a list containing configuration for paper dataset using lazy loading.

    Returns:
        List[DatasetConfig]: List with one dataset configuration.
    """
    datasets = [
            # ------------------------- without reduction
            DatasetConfig(name='hayes-roth', data_source=openml('hayes-roth', version=2)),
            DatasetConfig(name='newton_hema', data_source=openml('newton_hema', version=2)),
            DatasetConfig(name='iris', data_source=openml('iris')),
            DatasetConfig(name='grub-damage', data_source=openml('grub-damage', version=2)),
            DatasetConfig(name='servo', data_source=openml('servo')),
            DatasetConfig(name='wine', data_source=openml('wine')),
            DatasetConfig(name='seeds', data_source=openml('seeds')),
            DatasetConfig(name='glass', data_source=openml('glass')),
            DatasetConfig(name='conference_attendance', data_source=openml('conference_attendance')),
            DatasetConfig(name='prnn_synth', data_source=openml('prnn_synth')),
            DatasetConfig(name='heart-statlog', data_source=openml('heart-statlog')),
            DatasetConfig(name='heart-h', data_source=openml('heart-h', version=3)),
            DatasetConfig(name='cleveland', data_source=openml('cleveland')),
            DatasetConfig(name='diggle_table_a2', data_source=openml('diggle_table_a2', version=2)),
            DatasetConfig(name='ecoli', data_source=openml('ecoli')),
            DatasetConfig(name='Engine1', data_source=openml('Engine1')),

            # # ------------------------- reduce_dim
            DatasetConfig(name='leukemia', data_source=openml('leukemia'), reduce_dim=True, num_features=10),
            DatasetConfig(name='wisconsin', data_source=openml('wisconsin', version=2), reduce_dim=True, num_features=10),
            DatasetConfig(name='fri_c4_250_100', data_source=openml('fri_c4_250_100', version=2), reduce_dim=True, num_features=10),
            DatasetConfig(name='AP_Ovary_Lung', data_source=openml('AP_Ovary_Lung'), reduce_dim=True, num_features=10),
            DatasetConfig(name='spectf', data_source=openml('spectf'), reduce_dim=True, num_features=10),
            DatasetConfig(name='micro-mass', data_source=openml('micro-mass'), reduce_dim=True, num_features=10),
            DatasetConfig(name='AP_Endometrium_Breast', data_source=openml('AP_Endometrium_Breast'), reduce_dim=True, num_features=10),

            # # ------------------------- reduce_samples
            DatasetConfig(name='balance-scale', data_source=openml('balance-scale'), reduce_samples=True, sample_percentage=0.5),
            DatasetConfig(name='breast-w', data_source=openml('breast-w'), reduce_samples=True, sample_percentage=0.45),
            DatasetConfig(name='pima-indians-diabetes', data_source=openml('diabetes'), reduce_samples=True, sample_percentage=0.4),
            DatasetConfig(name='tic-tac-toe', data_source=openml('tic-tac-toe'), reduce_samples=True, sample_percentage=0.35),
            DatasetConfig(name='xd6', data_source=openml('xd6'), reduce_samples=True, sample_percentage=0.35),
            DatasetConfig(name='vowel', data_source=openml('vowel', version=2), reduce_samples=True, sample_percentage=0.3),

            # # ------------------------- reduce_samples + reduce_dim
            DatasetConfig(name='AP_Breast_Kidney', data_source=openml('AP_Breast_Kidney'), reduce_samples=True, sample_percentage=0.5, reduce_dim=True, num_features=10),
            DatasetConfig(name='vehicle', data_source=openml('vehicle'), reduce_samples=True, sample_percentage=0.35, reduce_dim=True, num_features=10),
            #DatasetConfig(name='oh5.wc', data_source=openml('oh5.wc', version=1, as_frame=False), reduce_samples=True, sample_percentage=0.35, reduce_dim=True, num_features=10),
            DatasetConfig(name='eating', data_source=openml('eating'), reduce_samples=True, sample_percentage=0.35, reduce_dim=True, num_features=10),
            DatasetConfig(name='qsar-biodeg', data_source=openml('qsar-biodeg'), reduce_samples=True, sample_percentage=0.3, reduce_dim=True, num_features=10),
            DatasetConfig(name='cnae-9', data_source=openml('cnae-9'), reduce_samples=True, sample_percentage=0.3, reduce_dim=True, num_features=10),

            
            # # ------------------------- MedMNIST
            DatasetConfig(name='pathmnist', data_source=WrappedMedMNIST(name='pathmnist'), reduce_samples=True, sample_percentage=0.0028, reduce_dim=True, num_features=10),
            DatasetConfig(name='dermamnist', data_source=WrappedMedMNIST(name='dermamnist'), reduce_samples=True, sample_percentage=0.03, reduce_dim=True, num_features=10),
            DatasetConfig(name='octmnist', data_source=WrappedMedMNIST(name='octmnist'), reduce_samples=True, sample_percentage=0.0027, reduce_dim=True, num_features=10),
            DatasetConfig(name='bloodmnist', data_source=WrappedMedMNIST(name='bloodmnist'), reduce_samples=True, sample_percentage=0.0176, reduce_dim=True, num_features=10),
            DatasetConfig(name='tissuemnist', data_source=WrappedMedMNIST(name='tissuemnist'), reduce_samples=True, sample_percentage=0.0013, reduce_dim=True, num_features=10),
            DatasetConfig(name='organamnist', data_source=WrappedMedMNIST(name='organamnist'), reduce_samples=True, sample_percentage=0.0051, reduce_dim=True, num_features=10),
            DatasetConfig(name='organcmnist', data_source=WrappedMedMNIST(name='organcmnist'), reduce_samples=True, sample_percentage=0.0127, reduce_dim=True, num_features=10),
            DatasetConfig(name='organsmnist', data_source=WrappedMedMNIST(name='organsmnist'), reduce_samples=True, sample_percentage=0.0119, reduce_dim=True, num_features=10),
            
            
            
            
            
            # DatasetConfig(name='rabe_131', data_source=openml('rabe_131', version=2)),
            # DatasetConfig(name='zoo', data_source=openml('zoo')),
            # DatasetConfig(name='visualizing_environmental', data_source=openml('visualizing_environmental', version=2)),
            # DatasetConfig(name='breast-tissue', data_source=openml('breast-tissue', version=2)),
            
            # DatasetConfig(name='cmc', data_source=openml('cmc'), reduce_samples=True, sample_percentage=0.20),
            # DatasetConfig(name='wine-quality-red', data_source=openml('wine-quality-red'), reduce_samples=True, sample_percentage=0.2),
            # DatasetConfig(name='car', data_source=openml('car'), reduce_samples=True, sample_percentage=0.2),
            # DatasetConfig(name='car-evaluation', data_source=openml('car-evaluation'), reduce_samples=True, sample_percentage=0.2, reduce_dim=True, num_features=10),
            # DatasetConfig(name='digits', data_source=builtin(skdata.load_digits), reduce_samples=True, sample_percentage=0.2, reduce_dim=True, num_features=10),
            
            # DatasetConfig(name='led24', data_source=openml('led24'), reduce_samples=True, sample_percentage=0.15, reduce_dim=True, num_features=10),
            # DatasetConfig(name='thyroid-dis', data_source=openml('thyroid-dis'), reduce_samples=True, sample_percentage=0.15, reduce_dim=True, num_features=10),
            # DatasetConfig(name='steel-plates-fault', data_source=openml('steel-plates-fault', version=3), reduce_samples=True, sample_percentage=0.15, reduce_dim=True, num_features=10),
            # DatasetConfig(name='mfeat-karhunen', data_source=openml('mfeat-karhunen'), reduce_samples=True, sample_percentage=0.15, reduce_dim=True, num_features=10),            
            # DatasetConfig(name='satimage', data_source=openml('satimage'), reduce_samples=True, sample_percentage=0.05, reduce_dim=True, num_features=10),
            # DatasetConfig(name='spambase', data_source=openml('spambase'), reduce_samples=True, sample_percentage=0.07, reduce_dim=True, num_features=10),
            # DatasetConfig(name='har', data_source=openml('har'), reduce_samples=True, sample_percentage=0.03, reduce_dim=True, num_features=10),
            # DatasetConfig(name='PhishingWebsites', data_source=openml('PhishingWebsites'), reduce_samples=True, sample_percentage=0.03, reduce_dim=True, num_features=10),
            # DatasetConfig(name='eye_movements', data_source=openml('eye_movements'), reduce_samples=True, sample_percentage=0.03, reduce_dim=True, num_features=10),
            # DatasetConfig(name='pendigits', data_source=openml('pendigits'), reduce_samples=True, sample_percentage=0.03),
            # DatasetConfig(name='mnist_784', data_source=openml('mnist_784'), reduce_samples=True, sample_percentage=0.005, reduce_dim=True, num_features=10),
    ]

    return datasets


def load_medMNIST_datasets() -> List[DatasetConfig]:
    """
    Creates a list containing configurations for compatible MedMNIST datasets (single-label, multi-class) using lazy loading.

    Returns:
        List[DatasetConfig]: List of dataset configurations for compatible MedMNIST datasets.
    """
    datasets = [
        DatasetConfig(name='pathmnist', data_source=WrappedMedMNIST(name='pathmnist'), reduce_samples=True, sample_percentage=0.0028, reduce_dim=True, num_features=10),
        DatasetConfig(name='dermamnist', data_source=WrappedMedMNIST(name='dermamnist'), reduce_samples=True, sample_percentage=0.03, reduce_dim=True, num_features=10),
        DatasetConfig(name='octmnist', data_source=WrappedMedMNIST(name='octmnist'), reduce_samples=True, sample_percentage=0.0027, reduce_dim=True, num_features=10),
        DatasetConfig(name='bloodmnist', data_source=WrappedMedMNIST(name='bloodmnist'), reduce_samples=True, sample_percentage=0.0176, reduce_dim=True, num_features=10),
        DatasetConfig(name='tissuemnist', data_source=WrappedMedMNIST(name='tissuemnist'), reduce_samples=True, sample_percentage=0.0013, reduce_dim=True, num_features=10),
        DatasetConfig(name='organamnist', data_source=WrappedMedMNIST(name='organamnist'), reduce_samples=True, sample_percentage=0.0051, reduce_dim=True, num_features=10),
        DatasetConfig(name='organcmnist', data_source=WrappedMedMNIST(name='organcmnist'), reduce_samples=True, sample_percentage=0.0127, reduce_dim=True, num_features=10),
        DatasetConfig(name='organsmnist', data_source=WrappedMedMNIST(name='organsmnist'), reduce_samples=True, sample_percentage=0.0119, reduce_dim=True, num_features=10),
    ]
    return datasets
    
__all__ = [
    "NoiseType",
    "ClusterType",
    "DatasetConfig",
    "ExperimentConfig",
    "NoiseConfig",
    "OverviewExperimentConfig",
    "ClusteringConfig",
    "load_all_datasets",
    "load_one_dataset",
    "load_tree_datasets",
    "load_small_datasets",
    "load_small_datasets_not_reduced",
    "load_paper_datasets",
    "load_medMNIST_datasets",
    "openml",
    "builtin",
]

@dataclass
class OverviewExperimentConfig:
    def __init__(self, dataset_name, n, m, c, nn):
        self.dataset_name = dataset_name
        self.n_samples = n
        self.m_features = m
        self.c_classes = c
        self.suggested_knn = nn