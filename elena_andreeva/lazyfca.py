__author__ = 'lena'

from multiprocessing import Value, Lock
from Queue import Queue
import datetime
from threading import Thread
import sklearn.preprocessing as preprocessing
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import accuracy_score

debug = False
pos_intes = dict()
neg_inters = dict()


class Counter(object):
    def __init__(self, initval=0):
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1

    def value(self):
        with self.lock:
            return self.val.value


def read_data(filename, positive_class_name):
    f = open(filename)
    attrs_set = []
    target = []
    for line in f:
            example = dict()
            values = line.rstrip("\n").split(",")
            for j in range(0, len(values)-1):
                example["f_" + str([j])]= values[j]
            if values[-1] == positive_class_name:
                target.append(1)
            else:
                target.append(0)
            attrs_set.append(example)
    return attrs_set, target


def rf_classify(dataset_name, iter_num, positive_class_name):
    training_set, target = read_data(dataset_name + "/train_" + str(iter_num) + ".csv", positive_class_name)
    vec = DictVectorizer()
    train = vec.fit_transform(training_set).toarray()
    clf = RandomForestClassifier(n_estimators=100)
    clf = clf.fit(train, target)
    test_set, target = read_data(dataset_name + "/test_" + str(iter_num) + ".csv", positive_class_name)
    test = vec.transform(test_set).toarray()
    pred = clf.predict(test)
    if debug:
        print ("Random Forest accuracy: " + str(accuracy_score(target, pred)))
    return accuracy_score(target, pred)


def bayes_classify(dataset_name, iter_num, positive_class_name):
    training_set, target = read_data(dataset_name + "/train_" + str(iter_num) + ".csv", positive_class_name)
    vec = DictVectorizer()
    train = vec.fit_transform(training_set).toarray()
    clf = BernoulliNB()
    clf = clf.fit(train, target)
    test_set, target = read_data(dataset_name + "/test_" + str(iter_num) + ".csv", positive_class_name)
    test = vec.transform(test_set).toarray()
    pred = clf.predict(test)
    if debug:
        print ("Bernoully accuracy: " + str(accuracy_score(target, pred)))
    return accuracy_score(target, pred)


def fca_supp_classify(positives, negatives, q, threshold, trues, falses):
    global pos_inters
    global neg_inters
    while q.unfinished_tasks > 0:
        item = q.get()
        pos_sum = 0.0
        neg_sum = 0.0
        for positive_example in positives:
            pos_cur = 0.0
            neg_cur = 0.0
            inter = positive_example & item[0]
            if len(inter) < threshold:
                continue
            if inter in pos_inters:
                pos_sum += pos_inters[inter]
                continue
            for pos in positives:
                if inter <= pos:
                    pos_cur += len(inter)
            # for neg in negatives:
            #     if inter <= neg:
            #         neg_cur += len(inter)
            pos_sum += pos_cur/len(positives)
            pos_inters[inter] = pos_cur/len(positives)
            #neg_sum += neg_cur/len(negatives)
        for negative_example in negatives:
            pos_cur = 0.0
            neg_cur = 0.0
            inter = negative_example & item[0]
            if len(inter) < threshold:
                continue
            if inter in neg_inters:
                neg_sum += neg_inters[inter]
                continue
            # for pos in positives:
            #     if inter <= pos:
            #         pos_cur += len(inter)
            for neg in negatives:
                if inter <= neg:
                    neg_cur += len(inter)
            #pos_sum += pos_cur/len(positives)
            neg_sum += neg_cur/len(negatives)
            neg_inters[inter] = neg_cur/len(negatives)

        rang = "undefined"
        #if pos_sum > neg_sum:
        #if max_pos > max_neg:
        if pos_sum > neg_sum:
            rang = 1
        if neg_sum > pos_sum:
        #if max_neg > max_pos:# :
            rang = 0
        if rang == item[1]:
            #print "TRUE " + rang + " " + str(pos_sum) + " " + str(neg_sum)
            trues.increment()
        else:
            #print "FALSE " + rang + " " + str(pos_sum) + " " + str(neg_sum)
            falses.increment()
        q.task_done()
    #print(str(datetime.datetime.now()) + " File: " + test_file_name +" thr: " + str(threshold) + " TRUE: " + str(trues) + " FALSE: " + str(falses))
    #return float(trues) / (trues + falses)

def fca_foreign_classify(positives, negatives, q, threshold, trues, falses, undefined):
    global pos_inters
    global neg_inters
    while q.unfinished_tasks > 0:
        item = q.get()
        poses = 0
        negs = 0
        for positive_example in positives:
            inter = positive_example & item[0]
            if len(inter) < threshold:
                continue
            if inter in pos_inters:
                if not pos_inters[inter]:
                    poses += 1
                    continue
                else:
                    continue
            stranger = False
            for neg in negatives:
                 if inter <= neg:
                     stranger = True
                     break
            if not stranger:
                poses += 1
                pos_inters[inter] = False
            else:
                pos_inters[inter] = True
            #neg_sum += neg_cur/len(negatives)
        for negative_example in negatives:
            inter = negative_example & item[0]
            if len(inter) < threshold:
                continue
            if inter in neg_inters:
                if not neg_inters[inter]:
                    negs += 1
                    continue
                else:
                    continue
            stranger = False
            for pos in positives:
                if inter <= pos:
                    stranger = True
                    break
            if not stranger:
                negs += 1
                neg_inters[inter] = False
            else:
                neg_inters[inter] = True




        rang = "undefined"
        #if pos_sum > neg_sum:
        #if max_pos > max_neg:
        classified = 0
        if poses > negs:
            rang = 1
            classified += 1
        if negs >= poses:
        #if max_neg > max_pos:# :
            rang = 0
            classified += 1
        if negs + poses == 0:
            rang = 0
        if rang == item[1]:
            #print "TRUE " + rang + " " + str(pos_sum) + " " + str(neg_sum)
            trues.increment()
        else:
            #print "FALSE " + rang + " " + str(pos_sum) + " " + str(neg_sum)
            falses.increment()
        if classified == 0:
            undefined.increment()
        q.task_done()
    #print(str(datetime.datetime.now()) + " File: " + test_file_name +" thr: " + str(threshold) + " TRUE: " + str(trues) + " FALSE: " + str(falses))
    #return float(trues) / (trues + falses)


def encode_data_from_file(filename):
    f = open(filename)
    data = []
    for line in f:
        values = line.rstrip("\n").split(",")
        data.append(values)
    enc = preprocessing.LabelEncoder()
    enc.fit_transform(data)
    return 1


def process_dataset(dataset_name, parts_qty, positive_class_name):
    ###FCA
    global pos_inters
    global neg_inters
    f = open(dataset_name + "/train_1.csv")
    fea_qty = len(f.readline().split(","))
    f.close()
    accuracy = [0] * (fea_qty - 1)
    times = [0] * (fea_qty - 1)
    accuracy2 = [0] * (fea_qty - 1)
    times2 = [0] * (fea_qty - 1)
    for i in range(1, parts_qty + 1):
        f = open(dataset_name + "/train_" + str(i) + ".csv")
        positives = []
        negatives = []
        for line in f:
            example = []
            values = line.rstrip("\n").split(",")
            for j in range(0, len(values) -1):
                example.append("f_" + str([j]) + "_" + values[j])
            if values[-1] == positive_class_name:
                positives.append(frozenset(example))
            else:
                negatives.append(frozenset(example))
        if debug:
            print(str(datetime.datetime.now()) + " Contexts are ready")
        f = open(dataset_name + "/test_"+str(i)+".csv")
        items = []
        for line in f:
            example = []
            values = line.rstrip("\n").split(",")
            for j in range(0, 9):
                example.append("f_" + str([j]) + "_" + values[j])
            if values[-1] == positive_class_name:
                target = 1
            else:
                target = 0
            items.append((frozenset(example), target))

        ###run first fca algorithm
        print(str(datetime.datetime.now()) + " Start classifying (support) " + str(len(items)) + " items")
        max_intersect = 0
        for j in range(1, len(positives[0])):
            time1 = time.time()
            trues = Counter(0)
            falses = Counter(0)
            queue = Queue()
            for item in items:
                queue.put(item)
            pos_inters = dict()
            neg_inters = dict()

            for t in range(0, 5):
                worker = Thread(target=fca_supp_classify, args=(positives, negatives, queue, j, trues, falses))
                worker.setDaemon(False)
                worker.start()

            queue.join()
            time2 = time.time()
            accuracy[j] += float(trues.value()) / (trues.value() + falses.value())
            times[j] += time2 - time1
            if len(pos_inters) + len(neg_inters) == 0:
                break
            max_intersect = j

            print(str(datetime.datetime.now()) + " File: " + "test"+str(i)+".csv" + " thr: " + str(j) + " TRUE: " + str(trues.value()) + " FALSE: " + str(falses.value()))

        ###run second fca algorithm
        print(str(datetime.datetime.now()) + " Start classifying (stranger) " + str(len(items)) + " items")
        for j in range(1, max_intersect + 1):
            time1 = time.time()
            trues = Counter(0)
            falses = Counter(0)
            undefined = Counter(0)
            queue = Queue()
            for item in items:
                queue.put(item)
            pos_inters = dict()
            neg_inters = dict()

            for t in range(0, 5):
                worker = Thread(target=fca_foreign_classify, args=(positives, negatives, queue, j, trues, falses, undefined))
                worker.setDaemon(False)
                worker.start()

            queue.join()
            time2 = time.time()
            accuracy2[j] += float(trues.value()) / (trues.value() + falses.value())
            times2[j] += time2 - time1
            if undefined.value() > 0:
                break
            print(str(datetime.datetime.now()) + " File: " + "test"+str(i)+".csv" + " thr: " + str(j) + " TRUE: " + str(trues.value()) + " FALSE: " + str(falses.value()))

    accuracy = [x / parts_qty for x in accuracy]
    accuracy2 = [x / parts_qty for x in accuracy2]
    print("=== Summary for dataset " + dataset_name + " ===")
    print("FCA-support:")
    print("Best threshold is " + str(accuracy.index(max(accuracy))) + ", accuracy: " + str(max(accuracy)) + ", time: " + str(times[accuracy.index(max(accuracy))]) + " s")
    print("FCA-stranger:")
    print("Best threshold is " + str(accuracy2.index(max(accuracy2))) + ", accuracy: " + str(max(accuracy2)) + ", time: " + str(times2[accuracy2.index(max(accuracy2))]) + " s")
    print(accuracy)
    print(times)
    print(accuracy2)
    print(times2)

    ###Bayes

    print("Naive Bayes:")
    b_accuracy = 0
    b_time = 0
    for i in range(1, parts_qty + 1):
        time1 = time.time()
        b_accuracy += bayes_classify(dataset_name, i, positive_class_name)
        time2 = time.time()
        b_time += (time2 - time1)
    b_accuracy /= parts_qty
    print("Accuracy: " + str(b_accuracy) + ", time: " + str(b_time))


    ###Random Forest
    print("Random Forest:")
    rf_accuracy = 0
    rf_time = 0
    for i in range(1, parts_qty + 1):
        time1 = time.time()
        rf_accuracy += rf_classify(dataset_name, i, positive_class_name)
        time2 = time.time()
        rf_time += (time2 - time1)
    rf_accuracy /= parts_qty

    print("Accuracy: " + str(rf_accuracy) + ", time: " + str(rf_time))


    return [accuracy, accuracy2]


def main():
    f = open("config")
    datasets = []
    accuracies = []
    for line in f:
        datasets.append(line.rstrip("\n").split(" "))

    for dataset in datasets:
        accuracies.extend(process_dataset(dataset[0], int(dataset[1]), dataset[2]))
        print(accuracies)

main()


