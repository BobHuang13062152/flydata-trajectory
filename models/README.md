Place your deep learning model weights here.

- For LSTM forecasting, save weights to: models/lstm_forecaster.pt
- Expected architecture: TinyLSTMForecast(input=2, hidden=64, layers=1) predicting delta(lat,lng).
- You can retrain your own and export state_dict via torch.save(model.state_dict(), 'models/lstm_forecaster.pt').
