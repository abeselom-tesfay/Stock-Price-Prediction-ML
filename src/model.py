from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    
    model.add(LSTM(units=50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units=60, activation='relu', return_sequences=True))
    model.add(Dropout(0.3))
    
    model.add(LSTM(units=80, activation='relu', return_sequences=True))
    model.add(Dropout(0.4))
    
    model.add(LSTM(units=120, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
