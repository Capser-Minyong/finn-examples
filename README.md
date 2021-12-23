## finn-examples - Dataflow QNN inference accelerator examples on FPGAs
### This is a forked repository for HYU CCE0063(SOC Design Methodology) term project

------------

Only tested for Alveo U250

How to run


    cd /build/finn
    ./run-docker.sh build_custom ./../mobilenet-v1_CIFAR10 

------------

Plz check models under directory below
/build/mobilenet-v1_CIFAR10/models/brevitas_output

I added 4 models for mobilenetv1 with configurations below

mobilenetv1_imagenet_w4a4_baseline.onnx - baseline from Xilinx, Imagenet dataset, 4bit quantization with brevitas In-house algorithm
mobilenetv1_cifar10_w4a4_baseline.onnx - CIFAR10 dataset, 4bit quantization with brevitas In-house algorithm
mobilenetv1_cifar10_w4a4_pact.onnx - CIFAR10 dataset, 4bit quantization with PACT(https://arxiv.org/pdf/1805.06085.pdf)
mobilenetv1_cifar10_w2a2_pact.onnx - CIFAR10 dataset, 2bit quantization with PACT
mobilenetv1_cifar10_w2a2_pact_batchnorm_fused.onnx - Experimental, Don't use

you can choose model to run by modifying /build/mobilenet-v1_CIFAR10/build.py

    #choose name of model 
    #model_name = "mobilenetv1_imagenet_w4a4_baseline"
    #model_name = "mobilenetv1_cifar10_w4a4_baseline"
    #model_name = "mobilenetv1_cifar10_w4a4_pact"
    model_name = "mobilenetv1_cifar10_w2a2_pact"
    #model_name = "mobilenetv1_cifar10_w2a2_pact_batchnorm_fused"

------------

I modified /build/mobilenet-v1_CIFAR10/build.py so that you also can do pre-process(UINT8 Input), post-process(TopK), Tidy

If you want to build steps after pre_post or streamlined, modify build.py below

    #ONNX exported from Brevitas
    model_file = "models/brevitas_output/%s.onnx" % model_name
    #If you want to use ONNX already pro_post_processed or streamline, use this instead 
    #model_file = "models/before_streamline/%s_5_pre_post.onnx" % model_name
    #model_file = "models/after_streamline%s_6_streamlined.onnx" % model_name

    elif platform in alveo_platforms:
        return [
            "step_tidy_up",
            #step_pre_post = preprocess layer(Input UINT8 quantization) postprocess layer(Top-1 node at the end) + tidy_up
            step_pre_post,
            step_mobilenet_streamline,
            step_mobilenet_lower_convs,
            step_mobilenet_convert_to_hls_layers,
            "step_create_dataflow_partition",
            #enable it when you set folding based on target fps
            #"step_target_fps_parallelization",
            ########################################################
            #enable it when you set folding based on folding config
            "step_apply_folding_config",
            ########################################################
            "step_generate_estimate_reports",
            "step_hls_codegen",
            "step_hls_ipgen",
            "step_set_fifo_depths",
            step_mobilenet_slr_floorplan,
            "step_synthesize_bitfile",
            "step_make_pynq_driver",
            "step_deployment_package",
        ]

------------

If you want to change platform, modify below(I only tested for Alveo U250)

    # which platforms to build the networks for
    #zynq_platforms = ["ZCU102", "ZCU104"]
    zynq_platforms = [] # We only use Alveo U250
    #alveo_platforms = ["U50", "U200", "U250", "U280"]
    alveo_platforms = ["U250"] # We only use Alveo U250
    #platforms_to_build = zynq_platforms + alveo_platforms
    platforms_to_build = alveo_platforms # We only use Alveo U250

------------

Expremental : folding config vs target fps
If you want to use whether folding config or target fps, modify below
(I failed to acheive valid accuracy with target fps config)

    cfg = build_cfg.DataflowBuildConfig(
        steps=select_build_steps(platform_name),
        output_dir="output_%s_%s" % (model_name, release_platform_name),
        #enable it when you set folding based on target fps
        ################################################################################
        #auto_fifo_depths=True,
        #target_fps=20000,
        #mvau_wwidth_max=10000,
        ################################################################################
        #enable it when you set folding based on folding config
        ################################################################################
        # folding config comes with FIFO depths already
        auto_fifo_depths=False,
        folding_config_file="folding_config/%s_folding_config_cifar10.json" % platform_name,
        #folding_config_file="folding_config/%s_folding_config)imagenet.json" % platform_name,
        ###############################################################################
        synth_clk_period_ns=select_clk_period(platform_name),
        board=platform_name,
        shell_flow_type=shell_flow_type,
        vitis_platform=vitis_platform,
        # enable extra performance optimizations (physopt)
        vitis_opt_strategy=build_cfg.VitisOptStrategyCfg.PERFORMANCE_BEST,
        generate_outputs=[
            build_cfg.DataflowOutputType.PYNQ_DRIVER,
            build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
            build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
            build_cfg.DataflowOutputType.BITFILE,
            build_cfg.DataflowOutputType.DEPLOYMENT_PACKAGE,
        ],
    )

------------
