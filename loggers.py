import os
import re
import csv
import datetime
import pandas as pd
import matplotlib.pyplot as plt


class CSVLogger(object):

    def __init__(self, logger_path='logger.csv'):
        self.logger_path = logger_path

    def log(self, epoch, total_epochs, step, total_steps, iter_type, loss, accuracy_per_joint, average_accuracy):
        mode = 'a' if os.path.exists(self.logger_path) else 'w'

        with open(self.logger_path, mode) as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_NONE)

            if mode == 'w':
                writer.writerow([
                    'Time',
                    'Type',
                    'Epoch',
                    'Step',
                    'Loss',
                    'Average_accuracy',
                    'R_Ankle',
                    'R_Knee',
                    'R_Hip',
                    'L_Hip',
                    'L_Knee',
                    'L_Ankle',
                    'Pelvis',
                    'Thorax',
                    'Upper_neck',
                    'Head_top',
                    'R_Wrist',
                    'R_Elbow',
                    'R_Shoulder',
                    'L_Shoulder',
                    'L_Elbow',
                    'L_Wrist'
                ])

            accuracy_per_joint.double()

            writer.writerow([
                datetime.datetime.now().strftime("%H:%M:%S_%d.%m.%Y"),
                iter_type,
                epoch,
                step,
                "{0:.8f}".format(loss.item()),
                "{0:.4f}".format(average_accuracy),
                "{0:.3f}".format(accuracy_per_joint[0].item()),
                "{0:.3f}".format(accuracy_per_joint[1].item()),
                "{0:.3f}".format(accuracy_per_joint[2].item()),
                "{0:.3f}".format(accuracy_per_joint[3].item()),
                "{0:.3f}".format(accuracy_per_joint[4].item()),
                "{0:.3f}".format(accuracy_per_joint[5].item()),
                "{0:.3f}".format(accuracy_per_joint[6].item()),
                "{0:.3f}".format(accuracy_per_joint[7].item()),
                "{0:.3f}".format(accuracy_per_joint[8].item()),
                "{0:.3f}".format(accuracy_per_joint[9].item()),
                "{0:.3f}".format(accuracy_per_joint[10].item()),
                "{0:.3f}".format(accuracy_per_joint[11].item()),
                "{0:.3f}".format(accuracy_per_joint[12].item()),
                "{0:.3f}".format(accuracy_per_joint[13].item()),
                "{0:.3f}".format(accuracy_per_joint[14].item()),
                "{0:.3f}".format(accuracy_per_joint[15].item())
            ])

    def plot_data(self, plot_avg_acc=True, plot_avg_acc_joint=True, plot_loss=True, plot_avg_time=True):
        df = pd.read_csv(self.logger_path)

        df = normalize_dataframe(df)

        mean_epoch_grouping = df.groupby('Epoch', as_index=False).mean()
        sum_epoch_grouping = df.groupby('Epoch', as_index=False).sum()

        import pdb
        pdb.set_trace()

        if plot_avg_acc:
            ax = plt.gca()
            mean_epoch_grouping['Average_accuracy'].plot(kind='line', x='epoch', y='avg_acc', grid=True, ylim=(0.0, 1.0), ax=ax)
            plt.show()

        if plot_avg_acc_joint:
            ax = plt.gca()
            mean_epoch_grouping['Head_top'].plot(kind='line', x='epoch', y='head_top_acc', grid=True, ylim=(0.0, 1.0), ax=ax)
            plt.show()

        if plot_loss:
            ax = plt.gca()
            mean_epoch_grouping['Loss'].plot(kind='line', x='epoch', y='loss', grid=True, ax=ax)
            plt.show()

        if plot_avg_time:
            ax = plt.gca()
            sum_epoch_grouping['Time'].plot(kind='line', x='epoch', y='avg_time', ax=ax)
            plt.show()




def normalize_dataframe(df):
    date_format = "%H:%M:%S_%d.%m.%Y"
    epoch_extract_regex = '\[(\d+)\/\d+\]'
    step_extract_regex = '\[(\d+)\/\d+\]'

    for index, row in df.iterrows():
        df.at[index, 'Time'] = int(datetime.datetime.strptime(row['Time'], date_format).timestamp())
        df.at[index, 'Epoch'] = int(re.search(epoch_extract_regex, row['Epoch']).group(1)) - 1
        df.at[index, 'Step'] = int(re.search(step_extract_regex, row['Step']).group(1))

    df['Time'] = pd.to_numeric(df['Time'])
    df['Epoch'] = pd.to_numeric(df['Epoch'])
    df['Step'] = pd.to_numeric(df['Step'])

    return df