#!/usr/bin/env python3

from nimbro_projection_double_sphere.node_projection_double_sphere import NodeProjectionDoubleSphere

import nimbro_utils.node as utils_node


def main():
    utils_node.start_and_spin_node(NodeProjectionDoubleSphere)


if __name__ == "__main__":
    main()
