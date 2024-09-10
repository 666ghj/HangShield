import os
import sys
import dpkt
import socket

from xgboost import train
workspace = sys.path[0]


class one_flow(object):
    def __init__(self, pkt_id, timestamp, direction, pkt_length):
        """
        初始化一个one_flow对象，用于表示一个网络流。
        这个类的主要作用是管理一个网络流的所有信息，包括流的基本属性和突发事件列表。流的更新通过update方法实现。
        """
        # 固定的属性
        self.pkt_id = pkt_id

        # 解析pkt_id获取IP和端口信息
        detailed_info = pkt_id.split("_")
        self.client_ip = detailed_info[0]  # 客户端IP
        self.client_port = int(detailed_info[1])  # 客户端端口
        self.outside_ip = detailed_info[2]  # 外部IP
        self.outside_port = int(detailed_info[3])  # 外部端口

        self.start_time = timestamp  # 流的开始时间
        # 可更新的属性
        self.last_time = timestamp  # 最后一个数据包的时间戳
        self.pkt_count = 1  # 数据包计数

        # 初始化一个 burst 列表，用于存储该流中的所有 burst
        self.burst_list = [one_burst(timestamp, direction, pkt_length)]

    def update(self, timestamp, direction, pkt_length):
        """
        更新one_flow对象的属性，用于处理新的数据包。

        参数:
        timestamp (float): 新数据包的时间戳。
        direction (int): 新数据包的方向，1表示从客户端到外部，-1表示从外部到客户端。
        pkt_length (int): 新数据包的长度（字节数）。

        说明:
        这个方法会更新该流的总数据包数（pkt_count）和最后一个数据包的时间戳（last_time）。
        如果新数据包的方向与前一个数据包不同，则创建一个新的one_burst对象并添加到burst_list中；
        否则，更新现有的one_burst对象的属性。
        """
        self.pkt_count += 1  # 更新数据包计数
        self.last_time = timestamp  # 更新最后时间

        # 如果新数据包的方向与前一个不同，创建新的 burst，否则更新现有的 burst
        if self.burst_list[-1].direction != direction:
            self.burst_list.append(one_burst(timestamp, direction, pkt_length))
        else:
            self.burst_list[-1].update(timestamp, pkt_length)



class one_burst(object):
    def __init__(self, timestamp, direction, pkt_length):
        """
        初始化一个one_burst对象，用于表示一个网络流中的突发事件（burst）。
        这个类的主要作用是管理一个突发事件的所有信息，包括方向、开始时间、数据包数量和总字节数。突发事件的更新通过update方法实现。
        """
        # 固定的属性
        self.direction = direction  # 突发事件的方向
        self.start_time = timestamp  # 突发事件的开始时间
        # 可更新的属性
        self.last_time = timestamp  # 最后一个数据包的时间戳
        self.pkt_count = 1  # 数据包计数
        self.pkt_length = pkt_length  # 突发事件的总字节数

    def update(self, timestamp, pkt_length):
        """
        更新one_burst对象的属性，用于处理新的数据包。

        参数:
        timestamp (float): 新数据包的时间戳。
        pkt_length (int): 新数据包的长度（字节数）。

        说明:
        这个方法会更新该突发事件的总数据包数（pkt_count）、总字节数（pkt_length）和最后一个数据包的时间戳（last_time）。
        """
        self.last_time = timestamp  # 更新最后时间

        self.pkt_count += 1  # 更新数据包计数
        self.pkt_length += pkt_length  # 累加数据包长度



def inet_to_str(inet):
    return socket.inet_ntop(socket.AF_INET, inet)


def get_burst_based_flows(pcap):
    """
    从pcap文件中获取基于突发的流量信息。该函数遍历pcap文件，解析出每个数据包的源IP、目标IP、源端口、目标端口等信息，并根据这些信息判断流量的方向和长度。如果一个数据包与之前的数据包具有相同的源IP、源端口、目标IP、目标端口，则认为这两个数据包属于同一个流量。

    Parameters:
    pcap (list): 一个包含时间戳和数据包内容的列表。

    Returns:
    list: 返回一个包含所有突发流量信息的列表，每个元素是一个one_flow对象，包含了流量的ID、时间戳、方向和长度等信息。

    Exceptions:
    如果在解析数据包时发生错误，会打印错误信息并跳过当前数据包。
    """
    # 初始化一个字典用于存储当前流量信息
    current_flows = dict()
    # 遍历pcap中的数据包
    for i, (timestamp, buf) in enumerate(pcap):
        try:
            # 尝试解析以太网帧
            eth = dpkt.ethernet.Ethernet(buf)
        except Exception as e:
            # 如果解析失败，打印错误并跳过当前循环
            print(e)
            continue

        # 检查数据是否为IP数据包，如果不是则尝试解析SLL（串行线路）帧
        if not isinstance(eth.data, dpkt.ip.IP):
            eth = dpkt.sll.SLL(buf)
            if not isinstance(eth.data, dpkt.ip.IP):
                continue

        # 获取IP数据包的信息
        ip = eth.data
        pkt_length = ip.len

        # 将IP地址转换为字符串格式
        src_ip = inet_to_str(ip.src)
        dst_ip = inet_to_str(ip.dst)

        # *这边我只要 TCP UDP 的信息，并且
        if not isinstance(ip.data, dpkt.tcp.TCP) and not isinstance(ip.data, dpkt.udp.UDP):
            continue
        if isinstance(ip.data, dpkt.dns.DNS):
            continue

        # 检查数据是否为TCP数据包，如果不是则跳过当前循环
        # if not isinstance(ip.data, dpkt.tcp.TCP):
        #     continue

        # *进行简单的数据清洗，删除组播之类的信息
        # 判断开头是不是224
        if src_ip.startswith('224') or dst_ip.startswith('224'):
            continue

        # 获取数据包的信息
        srcport = ip.data.sport
        dstport = ip.data.dport
        direction = None

        # 根据目标端口大小判断流量方向
        if srcport > dstport:
            direction = 1
            pkt_id = src_ip+"_"+str(srcport)+"_"+dst_ip+"_"+str(dstport)
        # 让方便设置标签
        else:
            direction = -1
            pkt_id = dst_ip+"_"+str(dstport)+"_"+src_ip+"_"+str(srcport)

        # if dstport == 443:
        #     direction = -1
        #     pkt_id = src_ip+"_"+str(srcport)+"_"+dst_ip+"_"+str(dstport)
        # elif srcport == 443:
        #     direction = 1
        #     pkt_id = dst_ip+"_"+str(dstport)+"_"+src_ip+"_"+str(srcport)
        # else:
        #     continue

        # 如果当前流量ID已存在于字典中，则更新流量信息；否则创建新的流量对象
        if pkt_id in current_flows:
            current_flows[pkt_id].update(timestamp, direction, pkt_length)
        else:
            current_flows[pkt_id] = one_flow(
                pkt_id, timestamp, direction, pkt_length)

    # 删除长度为1的流
    for flow_id in list(current_flows.keys()):
        # if current_flows[flow_id].pkt_count == 1:
        if len(current_flows[flow_id].burst_list) == 1:
            # print(current_flows[flow_id].client_ip,
            #       current_flows[flow_id].outside_ip)
            del current_flows[flow_id]

    # 返回字典中所有流量对象的列表
    return list(current_flows.values())


def get_flows(file):
    """
    读取并解析pcap文件，获取所有基于突发的流量。该函数接受一个文件路径作为参数，打开并读取该文件，然后使用dpkt库的pcap.Reader方法解析文件内容。最后，调用get_burst_based_flows函数获取所有基于突发的流量。

    Parameters:
    file (str): 一个待读取和解析的pcap文件的文件路径。

    Returns:
    list: 返回一个包含所有基于突发的流量的列表。
    """
    # *需要分别判断.pcap和.pcapng两种格式
    if file.endswith(".pcap"):
        # 使用with语句打开文件，确保文件在操作完成后能正确关闭
        with open(file, "rb") as input:
            # 创建一个dpkt.pcap.Reader对象，用于读取pcap文件
            pcap = dpkt.pcap.Reader(input)
            # 调用get_burst_based_flows函数处理pcap数据，获取所有流信息
            all_flows = get_burst_based_flows(pcap)
            # 返回所有流信息
            return all_flows
    elif file.endswith(".pcapng"):
        # 使用with语句打开文件，确保文件在操作完成后能正确关闭
        with open(file, "rb") as input:
            # 创建一个dpkt.pcapng.Reader对象，用于读取pcapng文件
            pcapng = dpkt.pcapng.Reader(input)
            # 调用get_burst_based_flows函数处理pcapng数据，获取所有流信息
            all_flows = get_burst_based_flows(pcapng)
            # 返回所有流信息
            return all_flows


def generate_sequence_data(all_files_flows, output_file, label_file):
    """
    生成序列数据。该函数接受所有文件流，输出文件和标签文件作为参数，对每个流进行处理，提取出特征和标签，并将它们写入到指定的文件中。

    Parameters:
    all_files_flows (list): 包含所有文件流的列表。
    output_file (str): 用于存储输出特征的文件路径。
    label_file (str): 用于存储输出标签的文件路径。

    Returns:
    无返回值。

    注意: 该函数会直接修改传入的all_files_flows列表，不会创建新的列表。
    """
    # 初始化输出特征和标签列表
    output_features = []
    output_labels = []

    # 遍历所有文件流
    for flow in all_files_flows:
        # 初始化一个流的特征列表
        one_flow = []

        # 获取客户端IP和外部IP
        client_ip = flow.client_ip
        outside_ip = flow.outside_ip

        # 生成标签，格式为"客户端IP-外部IP"
        label = client_ip + '-' + outside_ip

        # 遍历每个流的突发列表
        for index, burst in enumerate(flow.burst_list):
            # 如果不是第一个突发，计算累积值并添加到流特征列表中
            if index != 0:
                current_cumulative = one_flow[-1] + \
                    (burst.pkt_length * burst.direction)
                one_flow.append(current_cumulative)
            # 如果是第一个突发，直接添加突发长度乘以方向到流特征列表中
            else:
                one_flow.append(burst.pkt_length * burst.direction)

        # 将流特征列表中的数值转换为字符串
        one_flow = [str(value) for value in one_flow]

        # 将流特征列表用逗号连接成一行字符串
        one_line = ",".join(one_flow)

        # 将这一行字符串添加到输出特征列表中
        output_features.append(one_line)

        # 将标签添加到输出标签列表中
        output_labels.append(label)

    # 将输出特征和标签写入对应的文件
    write_into_files(output_features, output_file)
    write_into_files(output_labels, label_file)


def write_into_files(output_features, output_file):
    with open(output_file, "w") as write_fp:
        output_features = [value+"\n" for value in output_features]
        write_fp.writelines(output_features)


def main(input_dir, output_path):
    """
    遍历输入目录，提取出文件中的流信息。如果文件无法提取流信息或者流信息为空，则跳过该文件。最后将所有文件的流信息合并，生成序列数据。

    Parameters:
    input_dir (str): 输入目录的路径，用于查找所有后缀为suffix的文件。
    output_path (str): 输出文件的路径，用于存储生成的序列数据。

    Returns:
    无返回值。函数执行后，会在指定路径生成序列数据。

    注意：
    1. 如果某个文件无法提取流信息或者流信息为空，会打印错误信息并跳过该文件。
    2. 所有文件的流信息会被合并，生成一个包含所有流信息的列表。
    3. 生成的序列数据和对应的标签文件都会保存在指定的输出路径下。
    """
    # 遍历输入目录，获取所有后缀为suffix的文件路径
    pcap_filedir = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            pcap_filedir.append(os.path.join(root, file))

    # 提取出文件中的流信息
    files = pcap_filedir
    all_files_flows = []
    for file in files:
        # *初始化获取的流信息
        flows_of_file = False
        try:
            # 如果文件无法提取流信息或者流信息为空，则跳过该文件
            flows_of_file = get_flows(file)
        except Exception as e:
            print(e)
            pass
        if flows_of_file == False:  # 错误记录
            print(file, "Critical Error2")
            continue
        if len(flows_of_file) <= 0:
            continue
        # 将所有文件的流信息合并
        all_files_flows += flows_of_file

    # 生成序列数据和对应的标签文件
    generate_sequence_data(all_files_flows, output_path,
                           output_path + '_labels')


if __name__ == "__main__":
    # 首先读取全部种类
    with open('types.txt', 'r') as file:
        # 回车分割
        type_list = file.read().split()
    for data_type_id, data_type in enumerate(type_list):
        input_dir = "./pcap/"+data_type
        output_path = "feature/feature_"+data_type
        main(input_dir, output_path)
