import joblib
import json
import pandas as pd
import traceback

MODEL_PATH = r"e:\traffic-accident\traffic-accident\models\model.pkl"
METADATA_PATH = r"e:\traffic-accident\traffic-accident\models\model_metadata.json"

try:
    print('Loading metadata...')
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    FEATURES = metadata['dataset']['features']
    CATEGORICAL = metadata['dataset']['categorical_features']

    print('Loading model...')
    model = joblib.load(MODEL_PATH)
    print('Model loaded:', type(model))
    # If model is a sklearn Pipeline, print named steps and feature info
    try:
        if hasattr(model, 'named_steps'):
            print('Pipeline steps:', list(model.named_steps.keys()))
        # print feature names expected by the final estimator if available
        final = None
        if hasattr(model, 'named_steps'):
            # sklearn pipeline: last step
            final = list(model.named_steps.values())[-1]
        else:
            final = model

        if hasattr(final, 'feature_names_in_'):
            print('Final estimator.feature_names_in_:', list(final.feature_names_in_))
        elif hasattr(model, 'feature_names_in_'):
            print('Pipeline.feature_names_in_:', list(model.feature_names_in_))
    except Exception as e:
        print('Could not inspect model internals:', e)

    # Build sample input
    sample = {}
    for feat in FEATURES:
        fn = feat.lower()
        if feat in CATEGORICAL or fn in [c.lower() for c in CATEGORICAL]:
            if 'weather' in fn:
                sample[feat] = 'Rain'
            elif 'road' in fn:
                sample[feat] = 'arterial'
            else:
                sample[feat] = 'unknown'
        else:
            # numeric defaults
            if 'hour' in fn:
                sample[feat] = 12
            elif 'day' in fn or 'dow' in fn:
                sample[feat] = 3
            elif 'month' in fn:
                sample[feat] = 6
            elif 'year' in fn:
                sample[feat] = 2020
            elif 'speed' in fn:
                sample[feat] = 40
            elif 'age' in fn:
                sample[feat] = 30
            else:
                sample[feat] = 0

    df = pd.DataFrame([sample], columns=FEATURES)
    print('Sample input:')
    print(df.head().to_dict(orient='records'))

    # Predict
    try:
        proba = model.predict_proba(df)
        pred = model.predict(df)
        print('predict_proba:', proba)
        print('predict:', pred)
    except AttributeError:
        # model may not implement predict_proba (e.g., if a sklearn Pipeline without classifier)
        pred = model.predict(df)
        print('predict (no proba available):', pred)

except Exception as e:
    print('ERROR during model check:')
    traceback.print_exc()
