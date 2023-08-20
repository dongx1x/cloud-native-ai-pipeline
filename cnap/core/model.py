"""A Model module.

This module contains the definition of the ModelInfo class, which encapsulates
all relevant information about an Inference model in a machine learning pipeline.

In addition, ModelInfo instances can be easily serialized to dictionaries
for further processing or storage, by using Python's built-in `dict()` function.

Classes:
    ModelMetrics: A class that encapsulates the performance metrics of the inference model.
    ModelDetails: A class that encapsulates the details of the inference model.
    ModelInfo: A class for managing information about an Inference model.
    Model: A class for managing an Inference model.
"""

import requests
import uuid
from typing import Iterator, Tuple, Any
from datetime import datetime
from core.keybroker import AmberKeyBrokerClient

class ModelMetrics:
    """A class that encapsulates the performance metrics of the inference model.

    This class defines the performance metrics of the inference model, including
    accuracy, precision, recall, F1 score and loss value.

    Attributes:
        _accuracy (float): The accuracy of the model.
        _precision (float): The precision of the model.
        _recall (float): The recall of the model.
        _f1_score (float): The F1 score of the model.
        _loss (float): The loss value of the model.
    """

    def __init__(self,
                 accuracy: float,
                 precision: float,
                 recall: float,
                 f1_score: float,
                 loss: float):
        """Initialize a ModelMetrics object.

        This constructor initializes a ModelMetrics object with the given accuracy,
        precision, recall, F1 score and loss value.S

        Args:
            accuracy (float): The accuracy of the model.
            precision (float): The precision of the model.
            recall (float): The recall of the model.
            f1_score (float): The F1 score of the model.
            loss (float): The loss value of the model.
        """
        self._accuracy = accuracy
        self._precision = precision
        self._recall = recall
        self._f1_score = f1_score
        self._loss = loss

    @property
    def accuracy(self) -> float:
        """float: The accuracy of the model."""
        return self._accuracy

    @property
    def precision(self) -> float:
        """float: The precision of the model."""
        return self._precision

    @property
    def recall(self) -> float:
        """float: The recall of the model."""
        return self._recall

    @property
    def f1_score(self) -> float:
        """float: The F1 score of the model."""
        return self._f1_score

    @property
    def loss(self) -> float:
        """float: The loss value of the model."""
        return self._loss

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """The Iterator for ModelMetrics class."""
        yield 'accuracy', self._accuracy
        yield 'precision', self._precision
        yield 'recall', self._recall
        yield 'f1_score', self._f1_score
        yield 'loss', self._loss

class ModelDetails:
    """A class that encapsulates the details of the inference model.

    This class defines the details of the inference model, including
    name, version, framework, target and data type.

    Attributes:
        _name (str): The name of the model.
        _version (str): The version of the model.
        _framework (str): The framework used to train the model.
        _target (str): The target of the model.
        _dtype (str): The data type of the model.
    """

    def __init__(self, name: str, version: str, framework: str, target: str, dtype: str):
        """Initialize a ModelDetails object.

        This constructor initializes a ModelDetails object with the given name,
        version, framework, target and data type.

        Args:
            name (str): The name of the model.
            version (str): The version of the model.
            framework (str): The framework used to train the model.
            target (str): The target of the model.
            dtype (str): The data type of the model.
        """
        self._name = name
        self._version = version
        self._framework = framework
        self._target = target
        self._dtype = dtype

    @property
    def name(self) -> str:
        """str: The name of the model."""
        return self._name

    @property
    def version(self) -> str:
        """str: The version of the model."""
        return self._version

    @property
    def framework(self) -> str:
        """str: The framework used to train the model."""
        return self._framework

    @property
    def target(self) -> str:
        """str: The target of the model."""
        return self._target

    @property
    def dtype(self) -> str:
        """str: The data type of the model."""
        return self._dtype

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """The Iterator for ModelDetails class."""
        yield 'name', self._name
        yield 'version', self._version
        yield 'framework', self._framework
        yield 'target', self._target
        yield 'dtype', self._dtype

class ModelInfo:
    """A class for managing information about an Inference model.

    This class defines the information of the inference model, including
    ID, details, uploaded date and performance metrics.

    Attributes:
        _id (str): The ID of the model.
        _details (ModelDetails): The details of the model.
        _uploaded_date (datetime): The uploaded date of the model.
        _metrics (ModelMetrics): The performance metrics of the model.
    """

    def __init__(self,
                 id: str,
                 details: ModelDetails,
                 uploaded_date: datetime,
                 metrics: ModelMetrics):
        """Initialize a ModelInfo object.

        Args:
            details (ModelDetails): The details of the model.
            uploaded_date (datetime): The uploaded date of the model.
            metrics (ModelMetrics): The performance metrics of the model.
        """
        self._id = id
        self._details = details
        self._uploaded_date = uploaded_date
        self._metrics = metrics

    @property
    def id(self) -> str:
        """str: The ID of the model (string of UUID)."""
        return self._id

    @id.setter
    def id(self, new_str: str) -> None:
        """Set the ID of the model (string of UUID)."""
        self._id = str(uuid.UUID(new_str))

    @property
    def details(self) -> ModelDetails:
        """ModelDetails: The details of the model."""
        return self._details

    @property
    def uploaded_date(self) -> datetime:
        """datetime: The creation date of the model."""
        return self._uploaded_date

    @property
    def metrics(self) -> ModelMetrics:
        """ModelMetrics: The performance metrics of the model."""
        return self._metrics

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """The Iterator for ModelInfo class."""
        yield 'id', self.id
        yield 'details', dict(self._details)
        yield 'uploaded_date', self._uploaded_date
        yield 'metrics', dict(self._metrics)

class Model:
    """A class for managing an Inference model.

    This class defines the information of the inference model, including
    model information and model binary.

    Attributes:
        _model_info (ModelInfo): The information of the model.
        _model_binary (bytes): The binary of the model.
    """

    def __init__(self, server: str, model_id: str):
        """Initialize a Model object.

        Args:
            model_info (ModelInfo): The information of the model.
            model_binary (bytes): The binary of the model.
        Raises:
            ValueError: If model server or model id is None
        """
        if model_id is None or server is None:
            raise ValueError("Model server or model id can not be None")

        self._id = model_id
        self._server = server
        self._model_info = None
        self._model_binary = None
        self._path = None
        self.get_model_info()

    @property
    def path(self) -> str:
        """str: The path to save the model."""
        return self._path

    @property
    def model_info(self) -> ModelInfo:
        """ModelInfo: The information of the model."""
        return self._model_info

    @property
    def model_binary(self) -> bytes:
        """bytes: The binary of the model."""
        return self._model_binary

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """The Iterator for Model class."""
        yield 'model_info', dict(self._model_info)
        yield 'model_binary', self._model_binary

    def get_model_info(self):
        """Get model info from model server"""
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

        meta_data_url = self._server + '/' + self._id
        resp = requests.get(meta_data_url, headers=headers)
        if resp.status_code not in [200]:
            raise ConnectionError(f"Connect model server failed, error code: {resp.status_code}")
        meta_data = resp.json()
        # Get value from meta_data
        value = lambda key: None if key not in meta_data else meta_data[key]
        if value("id") is None or value("url") is None:
            raise RuntimeError("Model meta data is not valid")

        # Download model to /tmp
        resp = requests.get(value("url"))
        if resp.status_code not in [200]:
            raise ConnectionError(f"Download model from server failed, error code: {resp.status_code}")

        model_data = resp.content
        model_filename = value("url").split('/')[-1]
        model_filename = model_filename.replace(".enc", "").strip()
        self._path = "/tmp/" + model_filename

        if value("encrypted"):
            model_data = self.decrypt_model(value("kbs"), value("kbs_url"), value("key_id"), model_data)

        open(self._path, "wb").write(model_data)

        model_detail = ModelDetails(value("name"), value("version"), value("framework"),
                                    value("target"), value("dtype"))
        model_metrics = ModelMetrics(0, 0, 0, 0, 0)
        self._model_info = ModelInfo(value("id"), model_detail, 0, model_metrics)

    def decrypt_model(self, kbs, kbs_url, key_id, model_data):
        """Decrypt the model"""
        if kbs is None:
            ValueError("The KBS name/type can not be None")
        if kbs_url is None:
            ValueError("The KBS url can not be None")
        if key_id is None:
            ValueError("The key id can not be None")

        if kbs == "amber":
            amber_kbc = AmberKeyBrokerClient()
            key = amber_kbc.get_key(kbs_url, key_id)
            return amber_kbc.decrypt_data(model_data, key)
