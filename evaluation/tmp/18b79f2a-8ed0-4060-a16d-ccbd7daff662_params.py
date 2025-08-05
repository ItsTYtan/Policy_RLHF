datasets = [
    [
        dict(
            abbr='lukaemon_mmlu_virology',
            eval_cfg=dict(
                evaluator=dict(
                    type=
                    'opencompass.openicl.icl_evaluator.AccwithDetailsEvaluator'
                )),
            infer_cfg=dict(
                ice_template=dict(
                    template=dict(
                        A=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A\n',
                        B=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B\n',
                        C=
                        '{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C\n',
                        D='{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D\n'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.PPLInferencer'),
                prompt_template=dict(
                    ice_token='</E>',
                    template=dict(
                        A=
                        'The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: A',
                        B=
                        'The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: B',
                        C=
                        'The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: C',
                        D='The following are multiple choice questions (with answers) about  virology.\n\n</E>{input}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: D'
                    ),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    fix_id_list=[
                        0,
                        1,
                        2,
                        3,
                        4,
                    ],
                    type='opencompass.openicl.icl_retriever.FixKRetriever')),
            name='virology',
            path='opencompass/mmlu',
            reader_cfg=dict(
                input_columns=[
                    'input',
                    'A',
                    'B',
                    'C',
                    'D',
                ],
                output_column='target',
                train_split='dev'),
            type='opencompass.datasets.MMLUDataset'),
    ],
]
eval = dict(runner=dict(task=dict(dump_details=True)))
models = [
    dict(
        abbr='opt-125m_hf',
        batch_size=8,
        generation_kwargs=dict(),
        max_out_len=256,
        max_seq_len=None,
        model_kwargs=dict(),
        pad_token_id=None,
        path='facebook/opt-125m',
        peft_kwargs=dict(),
        peft_path=None,
        run_cfg=dict(num_gpus=1),
        stop_words=[],
        tokenizer_kwargs=dict(),
        tokenizer_path=None,
        type='opencompass.models.huggingface_above_v4_33.HuggingFaceBaseModel'
    ),
]
work_dir = 'outputs/default/20250731_081607'
