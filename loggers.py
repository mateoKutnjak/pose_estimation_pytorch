import os
import csv
import datetime
import pandas as pd


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
                "[" + str(epoch) + "/" + str(total_epochs) + "]",
                "[" + str(step) + "/" + str(total_steps) + "]",
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

    def plot_data(self, avg_acc=True, avg_acc_joint=True, loss=True, avg_time=True):
        df = pd.read_csv(self.logger_path)

        import pdb
        pdb.set_trace()
