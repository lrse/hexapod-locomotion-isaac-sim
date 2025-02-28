# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""
# This file was taken from Isaac Lab's repository to adapt the values to the hexapod, 
# which is about a 3rd of the size of an Anymal

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), #size of each sub-terrain
    border_width=10.0,#20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1, #TODO: hace falta reducir esto?
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.1),#(0.05, 0.23),
            step_width=0.2,#0.3,
            platform_width=1.0,#3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.1),#(0.05, 0.23),
            step_width=0.2,#0.3,
            platform_width=1.0,#3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            # proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            proportion=0.2, grid_width=0.3, grid_height_range=(0.01, 0.12), platform_width=1.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            # proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            proportion=0.1, slope_range=(0.0, 0.8), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            # proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            proportion=0.1, slope_range=(0.0, 0.8), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Rough terrains configuration."""

EXTRA_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,#20.0,
    num_rows=20, #Adding 10 extra levels
    num_cols=20,
    horizontal_scale=0.1, #TODO: hace falta reducir esto?
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.18),#(0.01, 0.14),#(0.01, 0.1),#(0.05, 0.23),
            step_width=0.2,#0.3,
            platform_width=1.0,#3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.18),#(0.01, 0.14),#(0.01, 0.1),#(0.05, 0.23),
            step_width=0.2,#0.3,
            platform_width=1.0,#3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            # proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            # proportion=0.2, grid_width=0.3, grid_height_range=(0.01, 0.12), platform_width=1.0
            # proportion=0.2, grid_width=0.3, grid_height_range=(0.01, 0.17), platform_width=1.0
            proportion=0.2, grid_width=0.3, grid_height_range=(0.01, 0.2), platform_width=1.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            # proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            # proportion=0.2, noise_range=(0.02, 0.14), noise_step=0.02, border_width=0.25
            proportion=0.2, noise_range=(0.02, 0.18), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            # proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            # proportion=0.1, slope_range=(0.0, 0.8), platform_width=2.0, border_width=0.25
            # proportion=0.1, slope_range=(0.0, 1.1), platform_width=2.0, border_width=0.25
            proportion=0.1, slope_range=(0.0, 1.5), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            # proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
            # proportion=0.1, slope_range=(0.0, 0.8), platform_width=2.0, border_width=0.25
            # proportion=0.1, slope_range=(0.0, 1.1), platform_width=2.0, border_width=0.25
            proportion=0.1, slope_range=(0.0, 1.5), platform_width=2.0, border_width=0.25
        ),
    },
)
"""Extra Challenging Rough terrains configuration."""

TEST_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(8.0, 8.0),
    border_width=10.0,#20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1, #TODO: hace falta reducir esto?
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.01, 0.2),#(0.05, 0.23),
        #     step_width=0.2,#0.3,
        #     platform_width=1.0,#3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.01, 0.2),#(0.05, 0.23),
            step_width=0.2,#0.3,
            platform_width=1.0,#3.0,
            border_width=1.0,
            holes=False,
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.3, grid_height_range=(0.01, 0.2), platform_width=1.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.2, slope_range=(0.0, 1.4), platform_width=2.0, border_width=0.25
        # ),
        # "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
        #     proportion=0.2, slope_range=(0.0, 0.8), platform_width=2.0, border_width=0.25
        # ),
    },
)


TEST_ONE_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(10.0, 10.0),
    border_width=10.0,#20.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1, #TODO: hace falta reducir esto?
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        # "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.075, 0.075),#(0.01, 0.2),
        #     step_width=0.2,#0.3,
        #     platform_width=1.0,#3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.075, 0.075),#(0.01, 0.2),
        #     step_width=0.2,#0.3,
        #     platform_width=1.0,#3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.3, grid_height_range=(0.1, 0.1), platform_width=1.0
        # ),
        # "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        #     proportion=0.2, noise_range=(0.02, 0.1), noise_step=0.02, border_width=0.25
        # ),
        # "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        #     proportion=0.2, slope_range=(0.5, 0.5), platform_width=2.0, border_width=0.25
        # ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.2, slope_range=(0.4, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)