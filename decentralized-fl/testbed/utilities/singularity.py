from .processes import *


def get_enodes(num_miners):
    ports = [8545]
    for i in range(num_miners):
        ports.append(8800 + i)

    # output = run_and_output(f"docker network inspect {network}")
    # containers = json.loads(output)[0]["Containers"]
    enodes = {}

    for port in ports:
        # ip = containers[container]["IPv4Address"]
        # ip = ip.split("/")[0]  # Remove CIDR part.

        enode = run_and_output(
            f'geth --exec "admin.nodeInfo.enode" attach http://127.0.0.1:{port}'
        )
        enode = enode.split("\n")[0]
        enode = enode.strip()
        # enode = enode.replace("127.0.0.1", ip)

        enodes[port] = enode

    print(enodes)

    return enodes


def connect_all(enodes):
    for port in enodes:
        for peer in enodes:
            print(
                run_and_output(
                    f"geth --exec \"admin.addPeer('{enodes[peer]}')\" attach http://127.0.0.1:{port}"
                )
            )
