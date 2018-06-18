import os
import re

train_file = "./cpbtrain.txt"
dev_file = "./cpbdev.txt"
test_file = "./cpbtest.txt"

A_dict_file = "./A_dict.txt"
B_dict_file = "./B_dict.txt"
C_dict_file = "./C_dict.txt"


def add_word_to_dict(_dict, _word):
    if _word in _dict:
        _dict[_word] += 1
    else:
        _dict[_word] = 1
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


def add_file_to_dict(_file, _dict_a, _dict_b, _dict_c):
    with open(_file, 'r') as f_i:
        for _line in f_i:
            _line_items = _line.strip().split(' ')
            for _item in _line_items:
                if _item == "":
                    continue
                _item_list = _item.split('/')
                flag = len(_item_list)
                _c = None
                if flag == 3:
                    [_a, _b, _c] = _item_list
                    _c = _c[_c.find('-')+1:]
                    _dict_c = add_word_to_dict(_dict_c, _c)
                    # print(_c)
                if flag == 2:
                    [_a, _b] = _item.split('/')
                _a = sub_word(_a)
                _dict_a = add_word_to_dict(_dict_a, _a)
                _dict_b = add_word_to_dict(_dict_b, _b)
    return _dict_a, _dict_b, _dict_c


def write_dict_in_file(_dict, _file, flag, max_num=None):
    word_list = sorted(_dict.items(), key=lambda d: d[1], reverse=True)
    with open(_file, 'w') as f_i:
        i = 0
        if flag == "A":
            f_i.write('_PAD\n')
            f_i.write('_UNK\n')
            i += 2
        if flag == "B":
            f_i.write('_PAD\n')
            i += 1
        if flag == "C":
            f_i.write('_PAD\n')
            i += 1

        for _item in word_list:
            if i < max_num:
                # f_i.write("%s:%d\n" % (_item[0], _item[1]))
                f_i.write("%s\n" % (_item[0]))
                i += 1
            else:
                break
    return


def main():
    A_dict = {}
    B_dict = {}
    C_dict = {}

    A_dict, B_dict, C_dict = add_file_to_dict(train_file, A_dict, B_dict, C_dict)
    A_dict, B_dict, C_dict = add_file_to_dict(dev_file, A_dict, B_dict, C_dict)
    A_dict, B_dict, C_dict = add_file_to_dict(test_file, A_dict, B_dict, C_dict)
    print("size: Word %d Pos %d Role %d"%(len(A_dict), len(B_dict), len(C_dict)))
    write_dict_in_file(A_dict, A_dict_file, flag="A", max_num=13000)
    write_dict_in_file(B_dict, B_dict_file, flag="B", max_num=10000)
    write_dict_in_file(C_dict, C_dict_file, flag="C", max_num=10000)


main()
