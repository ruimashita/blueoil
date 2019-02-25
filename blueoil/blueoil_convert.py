# -*- coding: utf-8 -*-
# Copyright 2018 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import os
import click
import subprocess
import shutil

from executor.export import run as run_export
from scripts.generate_project import run as run_generate_project

from lmnet.utils import executor, config as config_util
from lmnet import environment

from blueoil.vars import OUTPUT_TEMPLATE_DIR



def create_output_directory(output_root_dir, output_template_dir=None):
    """Create output directory from template."""

    template_dir = OUTPUT_TEMPLATE_DIR if not output_template_dir else output_template_dir
    # Recreate output_root_dir from template
    if os.path.exists(output_root_dir):
        shutil.rmtree(output_root_dir)
    shutil.copytree(template_dir, output_root_dir, symlinks=False, copy_function=shutil.copy)
    # Create output directories
    output_directories = get_output_directories(output_root_dir)
    for _, output_dir in output_directories.items():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    return output_directories


def get_output_directories(output_roor_dir):
    """Return output directories."""

    model_dir = os.path.join(output_roor_dir, "models")
    library_dir = os.path.join(model_dir, "lib")
    output_directories = dict(
        root_dir=output_roor_dir,
        model_dir=model_dir,
        library_dir=library_dir,
    )
    return output_directories


def strip_binary(output):
    """Strip binary file."""

    if output == "lm_x86.elf":
        subprocess.run(("strip", output))
    elif output == "lib_x86.so":
        subprocess.run(("strip", "-x", "--strip-unneeded", output))
    elif output in {"lm_arm.elf", "lm_fpga.elf"}:
        subprocess.run(("arm-linux-gnueabihf-strip", output))
    elif output in {"lib_arm.so", "lib_fpga.so"}:
        subprocess.run(("arm-linux-gnueabihf-strip", "-x", "--strip-unneeded", output))


def make_all(project_dir, output_dir):
    """Make each target."""

    make_list = [
        ["lm_x86", "lm_x86.elf"],
        # ["lm_arm", "lm_arm.elf"],
        ["lm_fpga", "lm_fpga.elf"],
        ["lib_x86", "lib_x86.so"],
        # ["lib_arm", "lib_arm.so"],
        ["lib_fpga", "lib_fpga.so"],
        # ["ar_x86", "libdlk_x86.a"],
        # ["ar_arm", "libdlk_arm.a"],
        # ["ar_fpga", "libdlk_fpga.a"],
    ]
    make_list = [
        ["lm_x86", "lm_x86.elf"],
        # ["lm_arm", "lm_arm.elf"],
        ["lm_fpga", "lm_fpga.elf"],
        # ["lib_x86", "lib_x86.so"],
        # ["lib_arm", "lib_arm.so"],
        # ["lib_fpga", "lib_fpga.so"],
        # ["ar_x86", "libdlk_x86.a"],
        # ["ar_arm", "libdlk_arm.a"],
        # ["ar_fpga", "libdlk_fpga.a"],
    ]
    running_dir = os.getcwd()
    # Change current directory to project directory
    os.chdir(project_dir)
    # os.environ["FLAGS"] = "-D__WITHOUT_TEST__"
    os.environ["CXXFLAGS"] = "-DFUNC_TIME_MEASUREMENT"
    # Make each target and move output files
    for target, output in make_list:
        subprocess.run(("make", "clean", "--quiet"))
        subprocess.run(("make",  target, "-j4",))
        strip_binary(output)
        output_file_path = os.path.join(output_dir, output)
        os.rename(output, output_file_path)
    # Return running directory
    os.chdir(running_dir)


def run(experiment_id, restore_path=None, output_template_dir=None):
    """Convert from trained model."""
    """
docker run \
	--rm \
	-it \
	--runtime=nvidia \
	-e PYTHONPATH=/home/blueoil:/home/blueoil/lmnet:/home/blueoil/dlk/python/dlk \
	-e OUTPUT_DIR=/home/blueoil/saved \
        -e CUDA_VISIBLE_DEVICES=0 \
	-v $(pwd)/config:/home/blueoil/config \
	-v $(pwd)/dataset:/home/blueoil/dataset \
	-v $(pwd)/lmnet/saved:/home/blueoil/saved \
	-v $(pwd)/result:/home/blueoil/result \
	-v $(pwd)/blueoil:/home/blueoil/blueoil \
	-v $(pwd)/lmnet/lmnet/networks:/home/blueoil/lmnet/lmnet/networks \
	-v $(pwd)/lmnet/executor:/home/blueoil/lmnet/executor \
	$(id -un)_blueoil:local_build \
       ./blueoil/cli.py convert -e person_segmentation/7_v5/

       ./blueoil/cli.py convert -e experiment
    """

    # Export model
    # run_export(experiment_id, restore_path, image_size=(None, None), images=[], config_file=None)

    # export_dir = run_export(experiment_id, restore_path, image_size=(192, 256), images=["lmnet/tests/fixtures/sample_images/cat.jpg"], config_file=None)
    export_dir = run_export(experiment_id, restore_path, image_size=(None, None), images=["lmnet/tests/fixtures/sample_images/cat.jpg"], config_file=None)

    # Set arguments
    input_pb_path = os.path.join(export_dir, "minimal_graph_with_shape.pb")
    dest_dir_path = export_dir
    project_name = "project"
    activate_hard_quantization = True
    threshold_skipping = True
    cache_dma = True
    debug = True
    # Generate project
    run_generate_project(
        input_path=input_pb_path,
        dest_dir_path=dest_dir_path,
        project_name=project_name,
        activate_hard_quantization=activate_hard_quantization,
        threshold_skipping=threshold_skipping,
        cache_dma=cache_dma,
        debug=debug,
    )

    # Create output dir from template
    output_root_dir = os.path.join(export_dir, "output")
    output_directories = create_output_directory(output_root_dir, output_template_dir)

    # Save meta.yaml to model output dir
    shutil.copy(os.path.join(export_dir, "meta.yaml"), output_directories.get("model_dir"))

    # Make
    project_dir_name = "{}.prj".format(project_name)
    project_dir = os.path.join(dest_dir_path, project_dir_name)
    make_all(project_dir, output_directories.get("library_dir"))

    return export_dir


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--experiment_id",
    help="id of this experiment.",
    required=True,
)
@click.option(
    "--restore_path",
    help="restore ckpt file base path. e.g. saved/experiment/checkpoints/save.ckpt-10001",
    default=None,
)
def main(experiment_id, restore_path):
    run(experiment_id, restore_path)


if __name__ == '__main__':
    main()
