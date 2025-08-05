datasets = [
    [
        dict(
            abbr='winograd',
            eval_cfg=dict(
                evaluator=dict(
                    type='opencompass.openicl.icl_evaluator.AccEvaluator')),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    template=dict({
                        0:
                        dict(round=[
                            dict(
                                prompt=
                                "{prompt} Q: In the previous text, what does '{pronoun}' refer to? A: {opt1}",
                                role='HUMAN'),
                        ]),
                        1:
                        dict(round=[
                            dict(
                                prompt=
                                "{prompt} Q: In the previous text, what does '{pronoun}' refer to? A: {opt2}",
                                role='HUMAN'),
                        ])
                    }),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            name='wsc285',
            path='winograd_wsc',
            reader_cfg=dict(
                input_columns=[
                    'prompt',
                    'pronoun',
                    'opt1',
                    'opt2',
                ],
                output_column='label',
                test_split='test',
                train_split='test'),
            trust_remote_code=True,
            type='opencompass.datasets.WinogradDataset'),
    ],
]
eval = dict(runner=dict(task=dict(dump_details=True)))
models = [
    dict(
        abbr='opt-125m-hf',
        batch_size=64,
        max_out_len=1024,
        path='facebook/opt-125m',
        run_cfg=dict(num_gpus=1),
        type='opencompass.models.HuggingFaceBaseModel'),
]
work_dir = 'outputs/default/20250731_082951'
