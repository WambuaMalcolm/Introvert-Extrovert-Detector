from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

cols_to_encode = ['Stage_fear', 'Drained_after_socializing']

cols_to_scale =["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]

def build_preprocessing():
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), cols_to_encode),
            ('num', StandardScaler(), cols_to_scale)
        ]
    )