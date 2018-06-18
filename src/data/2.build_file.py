import os
import re

train_file = "./cpbtrain.txt"
dev_file = "./cpbdev.txt"
test_file = "./cpbtest.txt"

A_dict_file = "./A_dict.txt"
B_dict_file = "./B_dict.txt"
C_dict_file = "./C_dict.txt"


train_path = "./train/"
dev_path = "./dev/"
test_path = "./test/"

a_path = 'a.txt'
b_path = 'b.txt'
c_path = 'c.txt'

a_id_path = 'a_id.txt'
b_id_path = 'b_id.txt'
c_id_path = 'c_id.txt'

max_seq_len = 0

def load_dict(_file):
    _dict = {}
    i = 0
    with open(_file, 'r') as f:
        for item in f:
            item = item.strip()
            if item != "":
                _dict[item] = i
                i += 1
    return _dict


def sub_word(_a):
    ans_a = _a
    if _a.find("年") != -1:
        if _a.find("一九")!=-1 or _a.find("二０")!=-1 or _a.find("１９")!=-1 \
                or _a.find("２０")!=-1 or _a.find("一八")!=-1 or _a.find("二零")!=-1\
                or _a.find("１８")!=-1:
            # print(_a.split())
            return '_YEAR'
    if _a.find("百分之")!=-1 or _a.find("％")!=-1:
        return '_PERCENT'

    if _a.find("ｗｗｗ·")!=-1:
        return '_NET'
    if _a[-1] == "万" or _a[-1] == "亿":
        if _a.find("一")!=-1 or _a.find("二")!=-1 or _a.find("三")!=-1 or _a.find("四")!=-1 or \
                        _a.find("五") != -1 or _a.find("六")!=-1 or _a.find("七")!=-1 or\
                        _a.find("八")!=-1 or _a.find("九")!=-1 or _a.find("十")!=-1 or \
                        _a.find("０")!=-1 or _a.find("１")!=-1 or _a.find("２")!=-1 or \
                        _a.find("３") != -1 or _a.find("４")!=-1 or _a.find("５")!=-1 or\
                        _a.find("６")!=-1 or _a.find("７")!=-1 or _a.find("８")!=-1 or\
                        _a.find("９")!=-1 or _a.find("两")!=-1 or _a.find("百")!=-1:
            return '_NUMBER'
    if (_a[-1]=="１" or _a[-1]=="２" or _a[-1]=="３" or _a[-1]=="４" or _a[-1]=="５" or
                _a[-1] == "６" or _a[-1] == "７" or _a[-1] == "８" or _a[-1] == "９" or _a[-1] == "０") and (_a[0] == "１" or _a[0] == "２" or _a[0] == "３" or _a[0] == "４" or _a[0] == "５" or
         _a[0] == "６" or _a[0] == "７" or _a[0] == "８" or _a[0] == "９" or _a[0] == "０"):
        return '_NUMBER'

    if _a.find("·")!=-1 and _a != "·":
        if not (_a.find("０") != -1 or _a.find("１") != -1 or _a.find("２") != -1 or \
                        _a.find("３") != -1 or _a.find("４") != -1 or _a.find("５") != -1 or \
                        _a.find("６") != -1 or _a.find("７") != -1 or _a.find("８") != -1 or \
                        _a.find("９") != -1):
            return '_NAME'

    if _a.find("月")!=-1:
        if _a.find("一") != -1 or _a.find("二") != -1 or _a.find("三") != -1 or _a.find("四") != -1 or \
                        _a.find("五") != -1 or _a.find("六") != -1 or _a.find("七") != -1 or \
                        _a.find("八") != -1 or _a.find("九") != -1 or _a.find("十") != -1 or \
                        _a.find("０") != -1 or _a.find("１") != -1 or _a.find("２") != -1 or \
                        _a.find("３") != -1 or _a.find("４") != -1 or _a.find("５") != -1 or \
                        _a.find("６") != -1 or _a.find("７") != -1 or _a.find("８") != -1 or \
                        _a.find("９") != -1:
            return '_MONTH'

    if _a[-1]=="分":
        if ((_a.find('时')!=-1)or(_a.find('点')!=-1)):
            if _a.find("一") != -1 or _a.find("二") != -1 or _a.find("三") != -1 or _a.find("四") != -1 or \
                            _a.find("五") != -1 or _a.find("六") != -1 or _a.find("七") != -1 or \
                            _a.find("八") != -1 or _a.find("九") != -1 or _a.find("十") != -1 or \
                            _a.find("０") != -1 or _a.find("１") != -1 or _a.find("２") != -1 or \
                            _a.find("３") != -1 or _a.find("４") != -1 or _a.find("５") != -1 or \
                            _a.find("６") != -1 or _a.find("７") != -1 or _a.find("８") != -1 or \
                            _a.find("９") != -1 or _a.find("两") != -1 or _a.find("百") != -1:
                return '_TIME'
    if _a[-1] == "日":
        if _a.find("０") != -1 or _a.find("１") != -1 or _a.find("２") != -1 or \
                        _a.find("３") != -1 or _a.find("４") != -1 or _a.find("５") != -1 or \
                        _a.find("６") != -1 or _a.find("７") != -1 or _a.find("８") != -1 or \
                        _a.find("９") != -1:
            return '_DAY'
        else:
            if _a.find("一") != -1 or _a.find("二") != -1 or _a.find("三") != -1 or _a.find("四") != -1 or \
                            _a.find("五") != -1 or _a.find("六") != -1 or _a.find("七") != -1 or \
                            _a.find("八") != -1 or _a.find("九") != -1 or _a.find("十") != -1:
                return '_DAY'
        # pass # pass # print(_a.split())

    p1=re.compile('^[零一二三四五六七八九十０１２３４５６７８９百千万亿多]*$')
    number = p1.match(_a)
    if number and _a != '百' and _a != '万' and _a != '亿' and _a != '多':
        return '_NUMBER'

    if _a.find('点')!=-1:
        if _a.find("一") != -1 or _a.find("二") != -1 or _a.find("三") != -1 or _a.find("四") != -1 or \
                        _a.find("五") != -1 or _a.find("六") != -1 or _a.find("七") != -1 or \
                        _a.find("八") != -1 or _a.find("九") != -1 or _a.find("十") != -1:
            return '_NUMBER'

    return ans_a


def build_data():
    # train
    A_lists = []
    B_lists = []
    C_lists = []
    with open(train_file, 'r') as f_i:
        for _line in f_i:
            _line_items = _line.strip().split(' ')
            A_list = []
            B_list = []
            C_list = []
            for _item in _line_items:
                if _item == "":
                    continue
                [_a, _b, _c] = _item.split('/')

                _c = _c[_c.find('-')+1:]

                A_list.append(_a)
                B_list.append(_b)
                C_list.append(_c)
                global max_seq_len
                max_seq_len = max(max_seq_len, len(A_list))

            A_lists.append(' '.join(A_list) + '\n')
            B_lists.append(' '.join(B_list) + '\n')
            C_lists.append(' '.join(C_list) + '\n')


    path = train_path
    with open(path + a_path, 'w') as f:
        f.writelines(A_lists)
    with open(path + b_path, 'w') as f:
        f.writelines(B_lists)
    with open(path + c_path, 'w') as f:
        f.writelines(C_lists)

    # dev
    A_lists = []
    B_lists = []
    C_lists = []
    with open(dev_file, 'r') as f_i:
        for _line in f_i:
            _line_items = _line.strip().split(' ')
            A_list = []
            B_list = []
            C_list = []
            for _item in _line_items:
                if _item == "":
                    continue
                [_a, _b, _c] = _item.split('/')

                _c = _c[_c.find('-') + 1:]

                A_list.append(_a)
                B_list.append(_b)
                C_list.append(_c)

            A_lists.append(' '.join(A_list) + '\n')
            B_lists.append(' '.join(B_list) + '\n')
            C_lists.append(' '.join(C_list) + '\n')

    path = dev_path
    with open(path + a_path, 'w') as f:
        f.writelines(A_lists)
    with open(path + b_path, 'w') as f:
        f.writelines(B_lists)
    with open(path + c_path, 'w') as f:
        f.writelines(C_lists)


    # test
    A_lists = []
    B_lists = []
    with open(test_file, 'r') as f_i:
        for _line in f_i:
            _line_items = _line.strip().split(' ')
            A_list = []
            B_list = []
            for _item in _line_items:
                if _item == "":
                    continue
                _item_list = _item.split('/')
                _a = _item_list[0]
                _b = _item_list[1]

                A_list.append(_a)
                B_list.append(_b)

            A_lists.append(' '.join(A_list) + '\n')
            B_lists.append(' '.join(B_list) + '\n')

    path = test_path
    with open(path + a_path, 'w') as f:
        f.writelines(A_lists)
    with open(path + b_path, 'w') as f:
        f.writelines(B_lists)


def build_id_data(A, B, C):
    # train
    A_lists = []
    B_lists = []
    C_lists = []
    with open(train_file, 'r') as f_i:
        for _line in f_i:
            _line_items = _line.strip().split(' ')
            A_list = []
            B_list = []
            C_list = []
            for _item in _line_items:
                if _item == "":
                    continue
                [_a, _b, _c] = _item.split('/')
                _a = sub_word(_a)
                if _a not in A:
                    _a = '_UNK'
                _c = _c[_c.find('-')+1:]

                A_list.append(str(A[_a]))
                B_list.append(str(B[_b]))
                C_list.append(str(C[_c]))

            A_lists.append(' '.join(A_list) + '\n')
            B_lists.append(' '.join(B_list) + '\n')
            C_lists.append(' '.join(C_list) + '\n')

    path = train_path
    with open(path + a_id_path, 'w') as f:
        f.writelines(A_lists)
    with open(path + b_id_path, 'w') as f:
        f.writelines(B_lists)
    with open(path + c_id_path, 'w') as f:
        f.writelines(C_lists)

    # dev
    A_lists = []
    B_lists = []
    C_lists = []
    with open(dev_file, 'r') as f_i:
        for _line in f_i:
            _line_items = _line.strip().split(' ')
            A_list = []
            B_list = []
            C_list = []
            for _item in _line_items:
                if _item == "":
                    continue
                [_a, _b, _c] = _item.split('/')
                _a = sub_word(_a)
                if _a not in A:
                    _a = '_UNK'
                _c = _c[_c.find('-') + 1:]

                A_list.append(str(A[_a]))
                B_list.append(str(B[_b]))
                C_list.append(str(C[_c]))

            A_lists.append(' '.join(A_list) + '\n')
            B_lists.append(' '.join(B_list) + '\n')
            C_lists.append(' '.join(C_list) + '\n')

    path = dev_path
    with open(path + a_id_path, 'w') as f:
        f.writelines(A_lists)
    with open(path + b_id_path, 'w') as f:
        f.writelines(B_lists)
    with open(path + c_id_path, 'w') as f:
        f.writelines(C_lists)


    # test
    A_lists = []
    B_lists = []
    C_lists = []
    with open(test_file, 'r') as f_i:
        for _line in f_i:
            _line_items = _line.strip().split(' ')
            A_list = []
            B_list = []
            C_list = []
            for _item in _line_items:
                if _item == "":
                    continue
                _item_list = _item.split('/')
                _a = _item_list[0]
                _b = _item_list[1]
                _c = 0
                _a = sub_word(_a)
                if _a not in A:
                    _a = '_UNK'
                if len(_item_list) == 3:
                    _c = C[_item_list[2]]
                A_list.append(str(A[_a]))
                B_list.append(str(B[_b]))
                C_list.append(str(_c))

            A_lists.append(' '.join(A_list) + '\n')
            B_lists.append(' '.join(B_list) + '\n')
            C_lists.append(' '.join(C_list) + '\n')

    path = test_path
    with open(path + a_id_path, 'w') as f:
        f.writelines(A_lists)
    with open(path + b_id_path, 'w') as f:
        f.writelines(B_lists)
    with open(path + c_id_path, 'w') as f:
        f.writelines(C_lists)


def main():
    A_dict = load_dict(A_dict_file)
    B_dict = load_dict(B_dict_file)
    C_dict = load_dict(C_dict_file)

    build_data()

    build_id_data(A_dict, B_dict, C_dict)

    print("max seq len = %d" % max_seq_len)

main()
