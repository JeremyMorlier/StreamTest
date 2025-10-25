import torch
from onnxruntime.training import artifacts

import onnx
from model.resnet18 import ResNet18
from onnx import shape_inference
from tools import apply_onnx_passes, run_stream_co

layer_stacks = [
    (0, 1),
    (2, 3),
    (4, 5, 6),
    (7, 8),
    (9, 10, 11),
    (12, 13),
    (14, 15, 16, 17),
    (18, 19),
    (20, 21, 22),
    (23, 24),
    (25, 26, 27, 28),
    (29, 30),
    (31, 32, 33),
    (34, 35),
    (36,),
    (37, 38, 39),
    (40,),
    (41,),
    (42,),
    (43,),
    (44, 45),
    (47,),
    (49,),
    (50,),
    (52, 53, 65),
    (54,),
    (61, 62),
    (66, 78),
    (67,),
    (74, 75),
    (80, 92),
    (81,),
    (88, 89),
    (93, 105),
    (94, 106, 115, 127),
    (101, 102),
    (111, 112),
    (116, 128, 140),
    (123, 124),
    (129, 142, 154),
    (136, 137),
    (143, 155, 167),
    (150, 151),
    (156, 168, 177, 189),
    (163, 164),
    (173, 174),
    (178, 190, 202),
    (185, 186),
    (191, 204, 216),
    (198, 199),
    (205, 217, 229),
    (212, 213),
    (218, 230, 239, 251),
    (225, 226),
    (235, 236),
    (240, 252, 264),
    (247, 248),
    (253, 266, 278),
    (260, 261),
    (267, 279, 291),
    (274, 275),
    (280, 295, 297, 298),
    (287, 288),
    (292,),
    (293,),
    (301, 302, 303, 304),
    (305, 306, 307, 308, 309, 310),
    (311, 312, 313, 314),
    (316, 317, 318, 319),
    (320, 321, 322, 323, 324, 325),
    (326, 327, 328, 329),
    (331, 332, 333, 334),
    (335, 336, 337, 338, 339, 340),
    (341, 342, 343, 344),
    (346, 347, 348, 349),
    (350, 351, 352, 353, 354, 355),
    (356, 357, 358, 359),
    (361, 362, 363, 364),
    (365, 366, 367, 368, 369, 370),
    (371, 372, 373, 374),
    (376, 377, 378, 379),
    (380, 381, 382, 383, 384, 385),
    (386, 387, 388, 389),
    (391, 392, 393, 394),
    (395, 396, 397, 398, 399, 400),
    (401, 402, 403, 404),
    (406, 407, 408, 409),
    (410, 411, 412, 413, 414, 415),
    (416, 417, 418, 419),
    (421, 422, 423, 424),
    (425, 426, 427, 428, 429, 430),
    (431, 432, 433, 434),
    (436, 437, 438, 439),
    (440, 441, 442, 443, 444, 445),
    (446, 447, 448, 449),
    (451, 452, 453, 454),
    (455, 456, 457, 458, 459, 460),
    (461, 462, 463, 464),
    (466, 467, 468, 469),
    (470, 471, 472, 473, 474, 475),
    (476, 477, 478, 479),
    (481, 482, 483, 484),
    (485, 486, 487, 488, 489, 490),
    (491, 492, 493, 494),
    (496, 497, 498, 499),
    (500, 501, 502, 503, 504, 505),
    (506, 507, 508, 509),
    (511, 512, 513, 514),
    (515, 516, 517, 518, 519, 520),
    (521, 522, 523, 524),
    (526, 527, 528, 529),
    (530, 531, 532, 533, 534, 535),
    (536, 537, 538, 539),
    (541, 542, 543, 544),
    (545, 546, 547, 548, 549, 550),
    (551, 552, 553, 554),
    (556, 557, 558, 559),
    (560, 561, 562, 563, 564, 565),
    (566, 567, 568, 569),
    (571, 572, 573, 574),
    (575, 576, 577, 578, 579, 580),
    (581, 582, 583, 584),
    (586, 587, 588, 589),
    (590, 591, 592, 593, 594, 595),
    (596, 597, 598, 599),
    (601, 602, 603, 604),
    (605, 606, 607, 608, 609, 610),
    (611, 612, 613, 614),
]

if __name__ == "__main__":
    folder = "results/resnet18_t/"
    onnx_path = f"{folder}/test.onnx"
    infered_path = f"{folder}/inferred.onnx"
    soc_path = "stream/stream/inputs/examples/hardware/tpu_like_quad_core.yaml"
    mapping_path = "stream/stream/inputs/examples/mapping/tpu_like_quad_core_fused.yaml"

    # Generate, Export and Infer Shapes of a ResNet18 Model
    model = ResNet18()
    torch_input = torch.randn(4, 3, 32, 32)
    torch.onnx.export(model, torch_input, onnx_path, opset_version=13)
    inferred_model = shape_inference.infer_shapes_path(onnx_path, infered_path)

    # Generate Backward
    base_model = onnx.load(infered_path)
    inits = base_model.graph.initializer
    requires_grad = []
    for init in inits:
        # if len(init.dims) != 1 :
        requires_grad.append(init.name)
    loss = artifacts.LossType(2)

    inferred_train_onnx_path4, forward_path, _, _ = apply_onnx_passes(
        base_model, None, folder, requires_grad, "onnx", check=False
    )

    errors = ""
    try:
        run_stream_co(
            inferred_train_onnx_path4,
            soc_path,
            mapping_path,
            id=0,
            output_path=folder,
            mode="lbl",
        )
    except Exception as e:
        print(e)
        errors += f"lbl_{e}\r\n"
    try:
        run_stream_co(
            inferred_train_onnx_path4,
            soc_path,
            mapping_path,
            id=1,
            output_path=folder,
            mode="fused",
            layer_stacks=[(i) for i in range(0, 614)],
        )
    except Exception as e:
        print("fused_lbl", e)
        errors += f"lbl_fused_{e}\r\n"
    for i, layer_stack in enumerate(layer_stacks):
        try:
            run_stream_co(
                inferred_train_onnx_path4,
                soc_path,
                mapping_path,
                id=i + 2,
                output_path=folder,
                mode="fused",
                layer_stacks=[layer_stack],
            )
        except Exception as e:
            print(layer_stack, e)
            errors += f"fused_{layer_stack}_{e}\r\n"

    print(errors)
