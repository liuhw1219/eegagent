bipolar_pairs = [
    ('FP1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('FP2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('A1', 'T3'), ('T3', 'C3'), ('C3', 'CZ'),
    ('CZ', 'C4'), ('C4', 'T4'), ('T4', 'A2'), 
    ('FP1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('FP2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
]

index2name = {i: f"{pair[0]}-{pair[1]}" for i, pair in enumerate(bipolar_pairs)}
index2name_ = {i: f"{pair[1]}-{pair[0]}" for i, pair in enumerate(bipolar_pairs)}
name2index = {**{v: k for k, v in index2name.items()},
              **{v: k for k, v in index2name_.items()}}


single_polar = [
    'Fp1', 'F3', 'C3', 'P3', 'O1',
    'F7', 'T3', 'T5', 'Fz', 'Fp2',
    'F4', 'C4', 'P4', 'O2', 'F8',
    'T4', 'T6', 'Cz', 'Pz'
]
single_index2name = {i: f"{name}" for i, name in enumerate(single_polar)}
single_name2index = {v: k for k, v in index2name.items()}


sleep_polar = ['Fpz-Cz', 'Pz-Oz']
sleep_index2name = {i: f"{name}" for i, name in enumerate(sleep_polar)}
sleep_name2index = {v: k for k, v in index2name.items()}


def montage_names(loader: str):
    """当前读取方式对应的导联名称顺序，必须与 dataLoad 里实际取道的顺序一致。"""
    if loader == "sleep":
        return list(sleep_polar)
    if loader == "mdd":
        return list(single_polar)
    return [f"{first}-{second}" for first, second in bipolar_pairs]


def set_montage(loader: str):
    """按读取方式切换 名称 ↔ 索引 映射。

    这里必须原地修改字典（clear/update），因为 tools 里是 `from .polar import name2index`，
    直接重新赋值不会同步到已经 import 的模块。
    """
    names = montage_names(loader)

    index2name.clear()
    index2name.update({index: name for index, name in enumerate(names)})

    index2name_.clear()
    index2name_.update({
        index: (f"{name.split('-')[1]}-{name.split('-')[0]}" if "-" in name else name)
        for index, name in enumerate(names)
    })

    name2index.clear()
    name2index.update({name: index for index, name in enumerate(names)})
    name2index.update({reversed_name: index for index, reversed_name in index2name_.items()})

    single_index2name.clear()
    single_index2name.update({index: name for index, name in enumerate(single_polar)})
    single_name2index.clear()
    single_name2index.update({name: index for index, name in enumerate(single_polar)})
    sleep_index2name.clear()
    sleep_index2name.update({index: name for index, name in enumerate(sleep_polar)})
    sleep_name2index.clear()
    sleep_name2index.update({name: index for index, name in enumerate(sleep_polar)})
