
import os

import numpy as np
import pandas as pd

import pickle

#LSTM with attention
import tensorflow as tf


from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, AdditiveAttention, Permute, Reshape, Multiply
from keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard, CSVLogger
from keras.models import load_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from eodhd import APIClient as eodhdClient

import yfinance as yf

import matplotlib.pyplot as plt
#================================================================================================
#
## This class defines and LSTM NN.
#
#================================================================================================
class WPLSTM:
    def __init__(self):
      # Configurable KERAS hyperparameters
      self._seq_length = 20
      self._batch_size = 64
      self._lstm_units = 50
      self._epochs = 100

    def get_ohlc_data(self, use_cache: bool = False) -> pd.DataFrame:
        ohlcv_file = "data/ohlcv.csv"

        if use_cache:
            if os.path.exists(ohlcv_file):
                return pd.read_csv(ohlcv_file, index_col=None)
            else:
                api = eodhdClient(os.environ['EODHD_API_KEY'])
                df = api.get_historical_data(
                    symbol="HSPX.LSE",
                    interval="d",
                    iso8601_start="2010-05-17",
                    iso8601_end="2023-10-04",
                )
                df.to_csv(ohlcv_file, index=False)
                return df
        else:
            api = eodhdClient(os.environ['EODHD_API_KEY'])
            return api.get_historical_data(
                symbol="HSPX.LSE",
                interval="d",
                iso8601_start="2010-05-17",
                iso8601_end="2023-10-04",
            )

    def create_sequences(self, data, seq_length):
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i : i + seq_length])
            y.append(data[i + seq_length, 3])  # The prediction target "close" is the 4th column (index 3)
        return np.array(x), np.array(y)

    def get_features(self, df: pd.DataFrame = None, feature_columns: list = ["open", "high", "low", "close", "volume"]) -> list:
        return df[feature_columns].values

    def get_target(self, df: pd.DataFrame = None, target_column: str = "close") -> list:
        return df[target_column].values

    def get_scaler(self, use_cache: bool = True) -> MinMaxScaler:
        scaler_file = "data/scaler.pkl"

        if use_cache:
            if os.path.exists(scaler_file):
                # Load the scaler
                with open(scaler_file, "rb") as f:
                    return pickle.load(f)
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                with open(scaler_file, "wb") as f:
                    pickle.dump(scaler, f)
                return scaler
        else:
            return MinMaxScaler(feature_range=(0, 1))

    def scale_features(self, scaler: MinMaxScaler = None, features: list = []):
        return scaler.fit_transform(features)

    def get_lstm_model(self, x_train, y_train, x_test, y_test, use_cache: bool = False) -> Sequential:
        model_file = "data/lstm_model.h5"

        if use_cache:
            if os.path.exists(model_file):
                # Load the model
                return load_model(model_file)
            else:
                # Train the LSTM model and save it
                model = Sequential()
                model.add(LSTM(units=self._lstm_units, activation='tanh', input_shape=(self._seq_length, 5)))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))

                model.compile(optimizer="adam", loss="mean_squared_error")
                model.fit(x_train, y_train, epochs=self._epochs, batch_size=self._batch_size, validation_data=(x_test, y_test))

                # Save the entire model to a HDF5 file
                model.save(model_file)

                return model

        else:
            # Train the LSTM model
            model = Sequential()
            model.add(LSTM(units=self._lstm_units, activation='tanh', input_shape=(self._seq_length, 5)))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(x_train, y_train, epochs=self._epochs, batch_size=self._batch_size, validation_data=(x_test, y_test))

            return model

    def evaluate_model(self, model, y_test, x_test: list = [], ) -> None:
        # Evaluate the model
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")
    
    def get_predicted_x_test_prices(self, model, scaler, x_test: np.ndarray = None):
        predicted = model.predict(x_test)

        # Create a zero-filled matrix to aid in inverse transformation
        zero_filled_matrix = np.zeros((predicted.shape[0], 5))

        # Replace the 'close' column of zero_filled_matrix with the predicted values
        zero_filled_matrix[:, 3] = np.squeeze(predicted)

        # Perform inverse transformation
        return scaler.inverse_transform(zero_filled_matrix)[:, 3]

    def plot_x_test_actual_vs_predicted(self, actual_close_prices: list = [], predicted_x_test_close_prices = []) -> None:
        # Plotting the actual and predicted close prices
        plt.figure(figsize=(14, 7))
        plt.plot(actual_close_prices, label="Actual Close Prices", color="blue")
        plt.plot(predicted_x_test_close_prices, label="Predicted Close Prices", color="red")
        plt.title("Actual vs Predicted Close Prices")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()

    def predict_next_close(self, model, df: pd.DataFrame = None, scaler: MinMaxScaler = None) -> float:
        # Take the last X days of data and scale it
        last_x_days = df.iloc[-self._seq_length:][["open", "high", "low", "close", "volume"]].values
        last_x_days_scaled = scaler.transform(last_x_days)

        # Reshape this data to be a single sequence and make the prediction
        last_x_days_scaled = np.reshape(last_x_days_scaled, (1, self._seq_length, 5))

        # Predict the future close price
        future_close_price = model.predict(last_x_days_scaled)

        # Create a zero-filled matrix for the inverse transformation
        zero_filled_matrix = np.zeros((1, 5))

        # Put the predicted value in the 'close' column (index 3)
        zero_filled_matrix[0, 3] = np.squeeze(future_close_price)

        # Perform the inverse transformation to get the future price on the original scale
        return scaler.inverse_transform(zero_filled_matrix)[0, 3]
    
    def run(self):
    
        # Retrieve 3369 days of S&P 500 data
        df = self.get_ohlc_data(use_cache=True)
        #print(df)

        features = self.get_features(df)
        target = self.get_target(df)

        scaler = self.get_scaler(use_cache=True)
        scaled_features = self.scale_features(scaler, features)

        x, y = self.create_sequences(scaled_features, self._seq_length)

        train_size = int(0.8 * len(x))  # Create a train/test split of 80/20%
        x_train, x_test = x[:train_size], x[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Re-shape input to fit lstm layer
        x_train = np.reshape(x_train, (x_train.shape[0], self._seq_length, 5))  # 5 features
        x_test = np.reshape(x_test, (x_test.shape[0], self._seq_length, 5))  # 5 features

        model = self.get_lstm_model(x_train, y_train, x_test, y_test, use_cache=True)

        # Evaluate the model
        self.evaluate_model(model, y_test, x_test)

        predicted_x_test_close_prices = self.get_predicted_x_test_prices(model, scaler, x_test)
        print("Predicted close prices:", predicted_x_test_close_prices)
        print(len(predicted_x_test_close_prices))
        
        analysis_status = "Completed"
        final_result = f"Predicted TEST next close price is: {predicted_x_test_close_prices}"

        # Plot the actual and predicted close prices for the test data
        # plot_x_test_actual_vs_predicted(df["close"].tail(len(predicted_x_test_close_prices)).values, predicted_x_test_close_prices)

        # Predict the next close price
        predicted_next_close =  self.predict_next_close(model, df, scaler)
        print("Predicted next close price:", predicted_next_close)
        
        analysis_status = "Completed"
        final_result = f"Predicted next close price is: {predicted_next_close}"
        
        return final_result


class WPLSTMAttention:
  
    def __init__(self):
      # Configurable Tensorflow hyperparameters
      self._seq_length = 60
      self._batch_size = 25
      self._lstm_units = 50
      self._epochs = 100
      
    def get_ohlc_data(self, use_cache: bool = False) -> pd.DataFrame:
        ohlcv_file = "data/ohlcv.csv"

        if use_cache:
            if os.path.exists(ohlcv_file):
                return pd.read_csv(ohlcv_file, index_col=None)
            else:
                api = eodhdClient(os.environ['EODHD_API_KEY'])
                df = api.get_historical_data(
                    symbol="HSPX.LSE",
                    interval="d",
                    iso8601_start="2020-01-01",
                    iso8601_end="2024-01-01",
                )
                df.to_csv(ohlcv_file, index=False)
                return df
        else:
            api = eodhdClient(os.environ['EODHD_API_KEY'])
            return api.get_historical_data(
                symbol="HSPX.LSE",
                interval="d",
                iso8601_start="2020-01-01",
                iso8601_end="2024-01-01",
            )

    def create_sequences(self, data, seq_length):
        #Choose a sequence length (like 60 days). This means, for every sample, the model will look at the last 60 days of data to make a prediction
        x, y = [], []
        for i in range(len(data) - seq_length):
            x.append(data[i : i + seq_length])
            y.append(data[i + seq_length, 3])  # The prediction target "close" is the 4th column (index 3)
        return np.array(x), np.array(y)

    def get_features(self, df: pd.DataFrame = None, feature_columns: list = ["open", "high", "low", "close", "volume"]) -> list:
        return df[feature_columns].values

    def get_target(self, df: pd.DataFrame = None, target_column: str = "close") -> list:
        return df[target_column].values

    def get_scaler(self, use_cache: bool = True) -> MinMaxScaler:
        scaler_file = "data/scaler.pkl"

        if use_cache:
            if os.path.exists(scaler_file):
                # Load the scaler
                with open(scaler_file, "rb") as f:
                    return pickle.load(f)
            else:
                scaler = MinMaxScaler(feature_range=(0, 1))
                with open(scaler_file, "wb") as f:
                    pickle.dump(scaler, f)
                return scaler
        else:
            return MinMaxScaler(feature_range=(0, 1))

    def scale_features(self, scaler: MinMaxScaler = None, features: list = []):
        return scaler.fit_transform(features)

    def get_lstm_model(self, x_train, y_train, x_test, y_test, use_cache: bool = False) -> Sequential:
        pass
        
        """
        model_file = "data/lstm_model.h5"

        if use_cache:
            if os.path.exists(model_file):
                # Load the model
                return load_model(model_file)
            else:
                # Train the LSTM model and save it
                model = Sequential()
                model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(seq_length, 5)))
                model.add(Dropout(0.2))
                model.add(Dense(units=1))

                model.compile(optimizer="adam", loss="mean_squared_error")
                model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

                # Save the entire model to a HDF5 file
                model.save(model_file)

                return model

        else:
            # Train the LSTM model
            model = Sequential()
            model.add(LSTM(units=lstm_units, activation='tanh', input_shape=(seq_length, 5)))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))

            model.compile(optimizer="adam", loss="mean_squared_error")
            model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

            return model
            """

    def evaluate_model(self, model, y_test, x_test: list = [], ) -> None:
        pass
        
        """
        # Evaluate the model
        y_pred = model.predict(x_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        print(f"Mean Squared Error: {mse}")
        print(f"Mean Absolute Error: {mae}")
        print(f"Root Mean Squared Error: {rmse}")
        
        """
    
    def get_predicted_x_test_prices(self, model, scaler, x_test: np.ndarray = None):
        pass
        
        """
        predicted = model.predict(x_test)

        # Create a zero-filled matrix to aid in inverse transformation
        zero_filled_matrix = np.zeros((predicted.shape[0], 5))

        # Replace the 'close' column of zero_filled_matrix with the predicted values
        zero_filled_matrix[:, 3] = np.squeeze(predicted)

        # Perform inverse transformation
        return scaler.inverse_transform(zero_filled_matrix)[:, 3]
        
        """

    def plot_x_test_actual_vs_predicted(self, actual_close_prices: list = [], predicted_x_test_close_prices = []) -> None:
        pass
        
        """
        
        # Plotting the actual and predicted close prices
        plt.figure(figsize=(14, 7))
        plt.plot(actual_close_prices, label="Actual Close Prices", color="blue")
        plt.plot(predicted_x_test_close_prices, label="Predicted Close Prices", color="red")
        plt.title("Actual vs Predicted Close Prices")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.show()
        
        """

    def predict_next_close(self, model, df: pd.DataFrame = None, scaler: MinMaxScaler = None) -> float:
        pass
        
        """
        # Take the last X days of data and scale it
        last_x_days = df.iloc[-seq_length:][["open", "high", "low", "close", "volume"]].values
        last_x_days_scaled = scaler.transform(last_x_days)

        # Reshape this data to be a single sequence and make the prediction
        last_x_days_scaled = np.reshape(last_x_days_scaled, (1, seq_length, 5))

        # Predict the future close price
        future_close_price = model.predict(last_x_days_scaled)

        # Create a zero-filled matrix for the inverse transformation
        zero_filled_matrix = np.zeros((1, 5))

        # Put the predicted value in the 'close' column (index 3)
        zero_filled_matrix[0, 3] = np.squeeze(future_close_price)

        # Perform the inverse transformation to get the future price on the original scale
        return scaler.inverse_transform(zero_filled_matrix)[0, 3]
        
        """

    def run(self):
        
        # Check TensorFlow version
        print("TensorFlow Version: ", tf.__version__)
        
        # Retrieve 3369 days of S&P 500 data
        df = self.get_ohlc_data(use_cache=True)
        #print(df)

        ##Section 1 - Data Preprocessing & preparation
        ##    Data Cleaning

        #checking for missing values
        missing_values = df.isnull().sum()
        print(f"Missing values: {missing_values}")

        # Filling missing values, if any
        df.fillna(method='ffill', inplace=True)

        # Handling Anomalies: Sometimes, datasets contain erroneous values due to glitches in data collection. 
        #     If you spot any anomalies (like extreme spikes in stock prices that are unrealistic), they should be corrected or removed.

        # Feature Selection
        # Deciding Features: For our model, we’ll use ‘Close’ prices, but you can experiment with additional features like ‘Open’, ‘High’, ‘Low’, and ‘Volume’.

        features = self.get_features(df)
        target = self.get_target(df)

        # Normalization
        #     Normalization is a technique used to change the values of numeric columns in the dataset to a common scale, without distorting differences in the ranges of values.
        #     Applying Min-Max Scaling: This scales the dataset so that all the input features lie between 0 and 1.
        scaler = self.get_scaler(use_cache=True)
        scaled_features = self.scale_features(scaler, features)

        # Creating Sequences
        #     LSTM models require input to be in a sequence format. We transform the data into sequences for the model to learn from.
        #     Defining Sequence Length: Choose a sequence length (like 60 days). This means, for every sample,
        #     the model will look at the last 60 days of data to make a prediction.

        X, y = self.create_sequences(scaled_features, self._seq_length)

        # Train-Test Split
        #    Split the data into training and testing sets to evaluate the model’s performance properly.
        #    Defining Split Ratio: Typically, 80% of data is used for training and 20% for testing.

        train_size = int(0.8 * len(X))  # Create a train/test split of 80/20%
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        #Reshaping Data for LSTM
        #    Finally, we need to reshape our data into a 3D format [samples, time steps, features] required by LSTM layers.

        
        X_train, y_train = np.array(X_train), np.array(y_train)
        ############ ValueError: cannot reshape array of size 227400 into shape (758,60,1)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

        # Section 2 - Building the LSTM with Attention Model
        #     Construction of our LSTM model with an added attention mechanism, tailored for predicting investment  stock patterns. 
        #     This requires TensorFlow and Keras.

        # Creating LSTM Layers
        #     Our LSTM model will consist of several layers, including LSTM layers for processing the time-series data. 
        #     The basic structure is as follows:

        model = Sequential()

        # Adding LSTM layers with return_sequences=True
        #     In this model, units represent the number of neurons in each LSTM layer. 
        #     return_sequences=True is crucial in the first layers to ensure the output includes sequences, which are essential for stacking LSTM layers. The final LSTM layer does not return sequences as we prepare the data for the attention layer.

        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=True))

        #Integrating the Attention Mechanism
        #    This custom layer computes a weighted sum of the input sequence, allowing the model to pay more attention to certain time steps.   
        #    The attention mechanism can be added to enhance the model’s ability to focus on relevant time steps:

        # Adding self-attention mechanism
        # The attention mechanism
        attention = AdditiveAttention(name='attention_weight')
        # Permute and reshape for compatibility
        model.add(Permute((2, 1)))
        model.add(Reshape((-1, X_train.shape[1])))
        attention_result = attention([model.output, model.output])
        multiply_layer = Multiply()([model.output, attention_result])
        # Return to original shape
        model.add(Permute((2, 1)))
        model.add(Reshape((-1, 50)))

        # Adding a Flatten layer before the final Dense layer
        model.add(tf.keras.layers.Flatten())

        # Final Dense layer
        model.add(Dense(1))

        #Optimizing the Model
        #     To enhance the model’s performance and reduce the risk of overfitting, we include Dropout and Batch Normalization.
        #     Dropout helps in preventing overfitting by randomly setting a fraction of the input units to 0 at each update during training,
        #     and Batch Normalization stabilizes the learning process.

        # Adding Dropout and Batch Normalization
        model.add(Dropout(0.2))
        model.add(BatchNormalization())

        #Model Compilation
        #    Finally, we compile the model with an optimizer and loss function suited for our regression task.
        #    adam optimizer is generally a good choice for recurrent neural networks, and mean squared error works well as a loss function for regression tasks like ours.

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Model Summary
        #    It’s beneficial to view the summary of the model to understand its structure and number of parameters.
        print(F"LSTM with Attention Model Summary: {model.summary()}")

        #Section 3: Training the Model
        #    Now that our LSTM model with attention is built, it’s time to train it using our prepared training set. 
        #    This process involves feeding the training data to the model and letting it learn to make predictions.

        # Assuming X_train and y_train are already defined and preprocessed
        #     Here, we train the model for 100 epochs with a batch size of 25.
        #     The validation_split parameter reserves a portion of the training data for validation, allowing us to monitor the model's performance on unseen data during training.

        history = model.fit(X_train, y_train, epochs=self._epochs, batch_size=self._batch_size, validation_split=0.2)

        #Overfitting and How to Avoid It
        #    Overfitting occurs when a model learns patterns specific to the training data, which do not generalize to new data. 
        #    Here are ways to avoid overfitting:
        #        - Validation Set: Using a validation set (as we did in the training code) helps in monitoring the model’s performance on unseen data.
        #        - Early Stopping: This technique stops training when the model’s performance on the validation set starts to degrade. Implementing early stopping in Keras is straightforward:

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        history = model.fit(X_train, y_train, epochs=self._epochs, batch_size=self._batch_size, validation_split=0.2, callbacks=[early_stopping])

        # Callback to save the model periodically
        model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

        # Callback to reduce learning rate when a metric has stopped improving
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)

        # Callback for TensorBoard
        tensorboard = TensorBoard(log_dir='./logs')

        # Callback to log details to a CSV file
        csv_logger = CSVLogger('training_log.csv')

        # Combining all callbacks
        callbacks_list = [early_stopping, model_checkpoint, reduce_lr, tensorboard, csv_logger]

        # Fit the model with the callbacks
        history = model.fit(X_train, y_train, epochs=self._epochs, batch_size=self._batch_size, validation_split=0.2, callbacks=callbacks_list)

        #Section 4: Evaluating Model Performance
        #    After training the model, the next step is to evaluate its performance using the test set. 
        #    This will give us an understanding of how well our model can generalize to new, unseen data.

        #Evaluating with the Test Set
        #    To evaluate the model, we first need to prepare our test data (X_test) in the same way we did for the training data. 
        #    Then, we can use the model's evaluate function:

        # Convert X_test and y_test to Numpy arrays if they are not already
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Ensure X_test is reshaped similarly to how X_train was reshaped
        # This depends on how you preprocessed the training data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Now evaluate the model on the test data
        test_loss = model.evaluate(X_test, y_test)
        print("Test Loss: ", test_loss)


        #Performance Metrics
        #    In addition to the loss, other metrics can provide more insights into the model’s performance. 
        #    For regression tasks like ours, common metrics include:
        #    - Mean Absolute Error (MAE): This measures the average magnitude of the errors in a set of predictions, without considering their direction.
        #    - Root Mean Square Error (RMSE): This is the square root of the average of squared differences between prediction and actual observation.

        #    To calculate these metrics, we can make predictions using our model and compare them with the actual values:
        
        # Making predictions
        y_pred = model.predict(X_test)

        # Calculating MAE and RMSE
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        print("Mean Absolute Error: ", mae)
        print("Root Mean Square Error: ", rmse)

        #Section 5: Predicting the Next 4 Candles
        #    Having trained and evaluated our LSTM model with an attention mechanism, the final step is to utilize it for predicting 
        #    the next 4 candles (days) of investment stock prices.
        
        #     Making Predictions
        #     To predict future stock prices, we need to provide the model with the most recent data points. 
        #     Let’s assume we have the latest 60 days of data prepared in the same format as X_train: and we want to predict the price for the next day:

        # Fetch the latest 60 days of AAPL stock data
        data = yf.download('HSPX.LSE', period='64d', interval='1d') # Fetch 64 days to display last 60 days in the chart

        # Select 'Close' price and scale it
        closing_prices = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(closing_prices)

        # Predict the next 4 days iteratively
        predicted_prices = []
        current_batch = scaled_data[-60:].reshape(1, 60, 1)  # Most recent 60 days

        for i in range(4):  # Predicting 4 days
            next_prediction = model.predict(current_batch)
            next_prediction_reshaped = next_prediction.reshape(1, 1, 1)
            current_batch = np.append(current_batch[:, 1:, :], next_prediction_reshaped, axis=1)
            predicted_prices.append(scaler.inverse_transform(next_prediction)[0, 0])

        # Creating a list of dates for the predictions
        last_date = data.index[-1]
        next_day = last_date + pd.Timedelta(days=1)
        prediction_dates = pd.date_range(start=next_day, periods=4)

        # Adding predictions to the DataFrame
        predicted_data = pd.DataFrame(index=prediction_dates, data=predicted_prices, columns=['Close'])

        # Combining both actual and predicted data
        combined_data = pd.concat([data['Close'], predicted_data['Close']])
        combined_data = combined_data[-64:] # Last 60 days of actual data + 4 days of predictions

        # Plotting the actual data
        plt.figure(figsize=(10,6))
        plt.plot(data.index[-60:], data['Close'][-60:], linestyle='-', marker='o', color='blue', label='Actual Data')

        # Plotting the predicted data
        plt.plot(prediction_dates, predicted_prices, linestyle='-', marker='o', color='red', label='Predicted Data')

        plt.title("AAPL Stock Price: Last 60 Days and Next 4 Days Predicted")
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

        
        return combined_data


