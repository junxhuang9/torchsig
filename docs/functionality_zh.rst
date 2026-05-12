TorchSig 功能代码说明与使用指南
================================

本文面向第一次阅读 TorchSig 代码的用户，按「这个仓库能做什么」「核心代码如何组织」「如何用代码生成、保存、加载数据」的顺序说明。TorchSig 是一个基于 PyTorch 数据管线的信号处理机器学习工具库，重点服务于射频（RF）和其它复数/实数信号的合成、增强、标注与模型训练数据准备。

能力总览
--------

TorchSig 的主要能力可以概括为以下几类：

* **合成信号生成**：内置 ASK/OOK、FSK/MSK/GFSK、PSK、QAM、OFDM、AM、FM、LFM/Chirp、Tone 等多类调制信号生成器。
* **窄带分类数据集**：通过 ``num_signals_max = 1`` 生成每个样本只包含一个主要信号的 I/Q 数据，适合调制识别或信号分类。
* **宽带检测数据集**：通过 ``num_signals_max > 1`` 在同一个噪声底上放置多个突发信号，并维护每个信号的时间/频率元数据，适合宽带检测任务。
* **信道与硬件损伤建模**：通过 ``Impairments`` 和独立 transform 模拟 AWGN、相位/频率偏移、漂移、IQ 不平衡、时钟抖动、衰落、阴影、非线性功放、量化、杂散、邻道/同道干扰等影响。
* **输入表示转换**：可输出原始复数 I/Q、二维 I/Q 张量、频谱图、频谱图图像，以及 YOLO 风格检测标签。
* **可复现实验**：``Seedable``、``WorkerSeedingDataLoader`` 和 YAML 配置支持固定随机种子，便于复现实验。
* **离线数据落盘与加载**：``DatasetCreator`` 支持把动态生成的数据写成静态数据集，默认 HDF5 存储；``StaticTorchSigDataset`` 支持随机访问已生成样本。
* **训练集成**：``TorchSigDataModule`` 提供 PyTorch Lightning DataModule 封装，支持生成/加载、切分 train/val/test、创建 DataLoader。

代码结构地图
------------

仓库中最常用的代码入口如下：

.. list-table:: 目录/文件职责
   :header-rows: 1
   :widths: 30 70

   * - 路径
     - 作用
   * - ``torchsig/datasets/datasets.py``
     - 定义 ``TorchSigIterableDataset`` 和 ``StaticTorchSigDataset``，负责动态合成样本和加载静态数据。
   * - ``torchsig/datasets/datamodules.py``
     - 定义 ``TorchSigDataModule``，面向 PyTorch Lightning 训练流程。
   * - ``torchsig/signals/``
     - 定义 ``Signal``、信号元数据、信号生成器基类，以及各类调制信号 builder。
   * - ``torchsig/signals/signal_lists.py``
     - 维护内置信号类别与调制族映射。
   * - ``torchsig/transforms/``
     - 定义基础 transform、信号 transform、损伤集合和元数据 transform。
   * - ``torchsig/utils/defaults.py``
     - 提供默认数据集元数据、默认 dataset 和默认 dataloader 的便捷函数。
   * - ``torchsig/utils/writer.py``
     - 定义 ``DatasetCreator``、collate 函数，以及写盘元信息检查逻辑。
   * - ``torchsig/utils/file_handlers/``
     - HDF5/NPY 等文件读写接口。
   * - ``torchsig/datasets/default_configs/``
     - 官方/示例 YAML 数据集配置。
   * - ``scripts/generate_dataset_from_config.py``
     - 从 YAML 配置生成离线数据集的命令行脚本。
   * - ``examples/``
     - Jupyter notebook 示例，覆盖数据生成、分类、检测、文件处理和 transform 可视化。

核心对象与数据流
----------------

TorchSig 的典型数据流是：

1. 准备数据集元数据，例如采样率、样本长度、FFT 大小、SNR 范围、信号个数范围和带宽范围。
2. 选择一个或多个信号生成器。``signal_generators="all"`` 会使用内置全部信号类别。
3. ``TorchSigIterableDataset`` 每次迭代时先生成噪声底，再按概率选择信号生成器生成 component signal。
4. 对单个 component signal 应用 ``component_transforms``，通常表示发射端或信号级损伤。
5. 将 component signal 随机频移并放置到完整样本的时间位置；多个信号会检查时频矩形重叠，并按 ``cochannel_overlap_probability`` 控制是否允许同道重叠。
6. 形成一个顶层 ``Signal``，其 ``data`` 是完整 I/Q 样本，``component_signals`` 保存每个子信号及元数据。
7. 对顶层样本应用 ``transforms``，例如接收端损伤、频谱图转换、二维 I/Q 转换或标签转换。
8. 根据 ``target_labels`` 返回 ``Signal``、裸 ``np.ndarray``，或 ``(data, label)`` 格式，直接交给 PyTorch DataLoader。
9. 如需离线复用，用 ``DatasetCreator`` 写盘；训练时用 ``StaticTorchSigDataset`` 或 ``TorchSigDataModule`` 加载。

信号类别能力
------------

``TorchSigSignalLists`` 将信号类别分成若干族。当前内置类别包括：

* **OOK/ASK**：``ook``、``4ask``、``8ask``、``16ask``、``32ask``、``64ask``。
* **FSK/MSK/GFSK/GMSK**：``2fsk`` 到 ``16fsk``，以及对应的 ``gfsk``、``msk``、``gmsk`` 变体。
* **PSK**：``bpsk``、``qpsk``、``8psk``、``16psk``、``32psk``、``64psk``。
* **QAM**：``16qam``、``32qam``、``32qam_cross``、``64qam``、``128qam_cross``、``256qam``、``512qam_cross``、``1024qam``。
* **OFDM**：``ofdm-64``、``ofdm-72``、``ofdm-128``、``ofdm-180``、``ofdm-256``、``ofdm-300``、``ofdm-512``、``ofdm-600``、``ofdm-900``、``ofdm-1024``、``ofdm-1200``、``ofdm-2048``。
* **模拟与其它**：``fm``、``am-dsb-sc``、``am-dsb``、``am-lsb``、``am-usb``、``lfm_data``、``lfm_radar``、``chirpss``、``tone``。

使用时可以传入单个族名、单个类别名、类别列表，或 ``"all"``。例如 ``signal_generators="fsk"`` 会查找 FSK 族生成器；``signal_generators=["bpsk", "qpsk", "16qam"]`` 只生成指定类别。

Transform 与损伤能力
--------------------

基础 transform 位于 ``torchsig/transforms/base_transforms.py``，包括：

* ``Compose``：顺序组合多个 transform。
* ``Lambda``：把自定义函数包装成 transform。
* ``Normalize``：归一化。
* ``RandomApply``：按概率应用一个 transform。
* ``RandAugment``：随机选择并应用增强。

信号 transform 位于 ``torchsig/transforms/transforms.py``，常见用途包括：

* **噪声/干扰**：``AWGN``、``AdditiveNoise``、``TimeVaryingNoise``、``AdjacentChannelInterference``、``CochannelInterference``、``Spurs``、``IntermodulationProducts``。
* **载波与时钟**：``CarrierFrequencyDrift``、``CarrierPhaseNoise``、``CarrierPhaseOffset``、``ClockDrift``、``ClockJitter``、``Doppler``。
* **通道/硬件**：``Fading``、``Shadowing``、``IQImbalance``、``NonlinearAmplifier``、``PassbandRipple``、``DigitalAGC``、``CoarseGainChange``、``Quantize``。
* **数据增强/形态变换**：``CutOut``、``PatchShuffle``、``RandomDropSamples``、``SpectralInversion``、``TimeReversal``、``ChannelSwap``。
* **表示转换**：``ComplexTo2D``、``InterleaveComplex``、``Spectrogram``、``SpectrogramImage``、``SpectrogramDropSamples``。

``Impairments`` 是预设损伤集合，会输出两组 transform：

* ``signal_transforms``：用于 component signal，通常描述单个信号在放置到样本前经历的影响。
* ``dataset_transforms``：用于完整样本，通常描述接收端、环境或后处理影响。

``metadata_transforms.py`` 中的 ``YOLOLabel`` 可把信号元数据转换成 YOLO 检测标签，常与 ``Spectrogram`` 一起用于宽带检测。

快速上手：生成一个内存数据集
----------------------------

下面示例生成一个小型窄带分类样本。``target_labels=["class_name"]`` 会让 dataset 返回 ``(data, label)``；如果设为 ``None``，则返回完整 ``Signal`` 对象，方便调试元数据。

.. code-block:: python

   from torchsig.datasets.datasets import TorchSigIterableDataset
   from torchsig.transforms.impairments import Impairments
   from torchsig.transforms.transforms import ComplexTo2D
   from torchsig.utils.defaults import TorchSigDefaults
   from torchsig.utils.data_loading import WorkerSeedingDataLoader

   metadata = TorchSigDefaults().default_dataset_metadata
   metadata.update({
       "num_iq_samples_dataset": 4096,
       "fft_size": 64,
       "fft_stride": 64,
       "num_signals_min": 1,
       "num_signals_max": 1,
   })

   impairments = Impairments(level=0)

   dataset = TorchSigIterableDataset(
       signal_generators=["bpsk", "qpsk", "16qam"],
       metadata=metadata,
       component_transforms=[impairments.signal_transforms],
       transforms=[impairments.dataset_transforms, ComplexTo2D()],
       target_labels=["class_name"],
       seed=123,
   )

   loader = WorkerSeedingDataLoader(
       dataset,
       seed=123,
       batch_size=8,
       num_workers=0,
   )

   batch = next(iter(loader))
   print(type(batch), len(batch))

窄带分类数据集配置要点
----------------------

窄带分类通常关注「一个样本对应一个信号类别」。建议配置：

* ``num_signals_min = 1`` 且 ``num_signals_max = 1``。
* ``signal_duration_in_samples_min`` 和 ``signal_duration_in_samples_max`` 接近样本长度，使信号覆盖大部分时间。
* ``target_labels`` 使用 ``["class_name"]`` 或 ``["class_index"]``。
* 输出可使用 ``ComplexTo2D`` 得到形状更适合神经网络的 I/Q 双通道表示。

.. code-block:: python

   from torchsig.datasets.datasets import TorchSigIterableDataset
   from torchsig.transforms.transforms import ComplexTo2D
   from torchsig.utils.defaults import TorchSigDefaults

   metadata = TorchSigDefaults().default_dataset_metadata
   metadata["num_signals_max"] = 1

   dataset = TorchSigIterableDataset(
       signal_generators="all",
       metadata=metadata,
       transforms=[ComplexTo2D()],
       target_labels=["class_index"],
       seed=7,
   )

   x, y = next(dataset)

宽带检测数据集配置要点
----------------------

宽带检测通常关注「一个宽带观测中包含多个信号及其时频位置」。建议配置：

* ``num_signals_max`` 设置为大于 1，例如 3、5 或更多。
* ``signal_duration_in_samples_min/max`` 设置为短于完整样本长度，形成突发信号。
* ``cochannel_overlap_probability`` 控制是否允许时频重叠。
* 使用 ``Spectrogram`` 将 I/Q 转换为频谱图。
* 使用 ``YOLOLabel`` 并设置 ``target_labels=["yolo_label"]`` 输出检测标签。

.. code-block:: python

   from torchsig.datasets.datasets import TorchSigIterableDataset
   from torchsig.transforms.metadata_transforms import YOLOLabel
   from torchsig.transforms.transforms import Spectrogram
   from torchsig.utils.defaults import TorchSigDefaults

   metadata = TorchSigDefaults().default_dataset_metadata
   metadata.update({
       "num_iq_samples_dataset": 262144,
       "fft_size": 512,
       "fft_stride": 512,
       "num_signals_min": 1,
       "num_signals_max": 5,
       "signal_duration_in_samples_min": 4096,
       "signal_duration_in_samples_max": 65536,
       "cochannel_overlap_probability": 0.2,
   })

   dataset = TorchSigIterableDataset(
       signal_generators="all",
       metadata=metadata,
       transforms=[
           Spectrogram(fft_size=metadata["fft_size"]),
           YOLOLabel(),
       ],
       target_labels=["yolo_label"],
       seed=11,
   )

   spectrogram, yolo_targets = next(dataset)

写入并加载离线数据集
--------------------

动态数据集适合在线增强；如果需要固定训练集、验证集或重复实验，可用 ``DatasetCreator`` 写盘，再用 ``StaticTorchSigDataset`` 加载。

.. code-block:: python

   from torchsig.datasets.datasets import StaticTorchSigDataset, TorchSigIterableDataset
   from torchsig.transforms.transforms import ComplexTo2D
   from torchsig.utils.data_loading import WorkerSeedingDataLoader
   from torchsig.utils.defaults import TorchSigDefaults
   from torchsig.utils.writer import DatasetCreator

   metadata = TorchSigDefaults().default_dataset_metadata
   metadata["num_iq_samples_dataset"] = 4096
   metadata["fft_size"] = 64
   metadata["fft_stride"] = 64

   dataset = TorchSigIterableDataset(
       signal_generators=["bpsk", "qpsk"],
       metadata=metadata,
       transforms=[ComplexTo2D()],
       target_labels=["class_name"],
       seed=2024,
   )

   loader = WorkerSeedingDataLoader(dataset, seed=2024, batch_size=16, num_workers=0)

   creator = DatasetCreator(
       dataloader=loader,
       dataset_length=100,
       root="./sample_dataset",
       overwrite=True,
       multithreading=False,
   )
   creator.create()

   static_dataset = StaticTorchSigDataset(
       root="./sample_dataset",
       target_labels=["class_name"],
   )

   x0, y0 = static_dataset[0]

使用 YAML 配置生成数据集
-----------------------

仓库提供了默认配置文件，例如 ``narrowband_toy_dataset.yaml``、``narrowband_clean_train_all.yaml``、``wideband_impaired_train_all.yaml`` 等。命令行脚本会读取 YAML，合并 ``TorchSigDefaults`` 的默认元数据，根据输出表示自动添加 ``ComplexTo2D`` 或 ``Spectrogram``/``YOLOLabel``。

示例命令：

.. code-block:: bash

   python scripts/generate_dataset_from_config.py \
       --root ./data \
       --config torchsig/datasets/default_configs/narrowband_toy_dataset.yaml \
       --overwrite \
       --batch_size 32 \
       --num_workers 0 \
       --save_config_copy

YAML 中几个重要字段：

.. list-table:: YAML 字段说明
   :header-rows: 1
   :widths: 30 70

   * - 字段
     - 说明
   * - ``dataset_id``
     - 输出目录名的一部分，脚本会写到 ``<root>/<dataset_id>``。
   * - ``dataset_length``
     - 要生成的样本数量。
   * - ``seed``
     - 随机种子。
   * - ``impairment_level``
     - 预设损伤等级。
   * - ``output.representation``
     - ``iq`` 或 ``spectrogram``。
   * - ``signal_sampling.mode``
     - ``per_signal`` 表示每个信号类别等概率；``per_family`` 表示每个调制族等概率。
   * - ``dataset_metadata``
     - 覆盖默认数据集元数据，例如样本长度、FFT、信号数量、SNR、带宽、频率范围。

使用 PyTorch Lightning DataModule
---------------------------------

如果训练代码使用 PyTorch Lightning，可以用 ``TorchSigDataModule`` 统一处理生成、落盘、加载和切分。

.. code-block:: python

   from torchsig.datasets.datamodules import TorchSigDataModule
   from torchsig.transforms.transforms import ComplexTo2D
   from torchsig.utils.defaults import TorchSigDefaults

   metadata = TorchSigDefaults().default_dataset_metadata
   metadata.update({
       "num_iq_samples_dataset": 4096,
       "fft_size": 64,
       "fft_stride": 64,
   })

   datamodule = TorchSigDataModule(
       root="./lightning_dataset",
       metadata=metadata,
       dataset_size=1000,
       dataset_splits=[0.7, 0.2, 0.1],
       batch_size=32,
       num_workers=0,
       transforms=[ComplexTo2D()],
       target_labels=["class_index"],
       impairment_level=0,
       overwrite=True,
       seed=99,
   )

   datamodule.prepare_data()
   datamodule.setup()
   train_loader = datamodule.train_dataloader()

常用元数据字段
--------------

.. list-table:: 数据集元数据速查
   :header-rows: 1
   :widths: 35 65

   * - 字段
     - 含义
   * - ``num_iq_samples_dataset``
     - 每个样本的 I/Q 点数。
   * - ``num_signals_min`` / ``num_signals_max``
     - 每个样本中放置的信号数量范围；1 通常是分类，大于 1 通常是检测。
   * - ``fft_size`` / ``fft_stride``
     - 频谱图计算使用的 FFT 大小和步长。
   * - ``sample_rate``
     - 采样率。
   * - ``noise_power_db``
     - 噪声底功率。
   * - ``snr_db_min`` / ``snr_db_max``
     - 单个信号的 SNR 随机范围。
   * - ``cochannel_overlap_probability``
     - 多信号场景中允许同道/时频重叠的概率。
   * - ``signal_duration_in_samples_min`` / ``signal_duration_in_samples_max``
     - 子信号持续时间范围。
   * - ``bandwidth_min`` / ``bandwidth_max``
     - 子信号带宽范围。
   * - ``signal_center_freq_min`` / ``signal_center_freq_max``
     - 子信号中心频率随机范围。
   * - ``frequency_min`` / ``frequency_max``
     - 数据集观测频率范围。

返回值与标签规则
----------------

``TorchSigIterableDataset`` 和 ``StaticTorchSigDataset`` 都会在应用 transform 后，根据 ``target_labels`` 决定返回内容：

* ``target_labels is None``：返回完整 ``Signal`` 对象，适合调试和继续访问元数据。
* ``target_labels=[]``：只返回 ``sample.data``。
* ``target_labels=["class_name"]``：返回 ``(sample.data, class_name)``。
* 多个标签，例如 ``["class_name", "snr_db"]``：返回 ``(sample.data, [class_name, snr_db])``。

对于 ``num_signals_max == 1`` 的分类任务，单个 component 的标签会被简化为标量；对于多信号检测任务，标签通常是列表或数组，每个 component signal 对应一个目标。

推荐学习路径
------------

1. 先运行 ``examples/create_dataset_example.ipynb``，理解如何生成和写盘。
2. 再阅读 ``torchsig/datasets/datasets.py`` 中 ``TorchSigIterableDataset.__generate_new_signal__``，理解噪声底、信号放置和元数据维护。
3. 查看 ``torchsig/signals/signal_lists.py`` 和 ``torchsig/signals/builders/``，了解可用调制类型及每种信号如何生成。
4. 查看 ``torchsig/transforms/transforms.py`` 和 ``examples/transforms/``，理解各类损伤/增强对数据的影响。
5. 使用 ``torchsig/datasets/default_configs/`` 中的 YAML 作为模板，创建自己的窄带分类或宽带检测配置。

实践建议
--------

* 调试新配置时，先把 ``num_iq_samples_dataset`` 和 ``dataset_length`` 设小，确认能快速生成样本。
* 分类任务优先用 ``target_labels=["class_index"]``，并固定 ``signal_generators`` 或记录 ``class_names``，保证训练/验证类别顺序一致。
* 检测任务先用 ``target_labels=None`` 返回 ``Signal``，检查 ``component_signals`` 的 ``center_freq``、``bandwidth``、``start_in_samples`` 等字段，再启用 ``YOLOLabel``。
* 多进程加载时使用 ``WorkerSeedingDataLoader`` 并传入 ``seed``，避免 worker 随机状态不可控。
* 需要可重复 benchmark 时，建议用 YAML + ``DatasetCreator`` 生成静态数据集，并保存原始配置。
