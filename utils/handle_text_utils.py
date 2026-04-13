

import os
import re
import pycountry
import geonamescache

project_root = os.path.join(os.path.dirname(__file__), "..") 



def get_first_few_sentences(text, min_word_num, max_word_num):
    """ textを文に分割して、最初の数文を連結して返す。連結したテキストの単語数がword_thresholdを超えないようにする。
    ~~ただし1文目ですでにword_thresholdを超える場合は、最初の1文だけを返す。~~
    1文目がword_thresholdを超える場合は、Noneを返す
    """
    if text is None:
        print("text is None")
        return None
    
    # 文末記号を保持して分割
    parts = re.split(r'(?<=[。．!?！？\n])\s*', text)
    # delimiters = re.findall(r'[。．!?！？]', text) -> 上のsplit方法では、文末記号もpartsに含まれるため、delimitersは必要ない
    # sentences = [s.strip() for s in parts if len(s) > 0]  # 空の文を除外
    
    word_count, char_count = 0, 0
    truncated_text = ""
    for i, sentence in enumerate(parts):
        sentence = sentence.strip()
        if len(sentence) == 0:
            # 空の文を除外
            continue

        word_count += len(sentence.split())  # 連結したテキストの単語数をカウント
        char_count += len(sentence)
        if word_count > max_word_num or char_count > max_word_num * 5:  # 一定の単語数・文字数を超えないようにする
            break
        truncated_text += sentence + " "
        # print(f"word count: {word_count}, char_count: {char_count}, text: {text[:200]}...")# , sentence: {sentence}")
    
    # return truncated_text if truncated_text != "" else None
    if word_count < min_word_num: # truncated_text == "" or word_count < min_word_num:
        # もし、1文目が既にmax_word_numを超えている、もしくは全文の単語数がmin_word_numより少ない場合は、Noneを返す
        print(f"skip.  word count: {word_count} < {}, char_count: {char_count}, {parts[0][:200]}...")# , sentence: {sentence}")
        return None
    else:
        # print(f"word count: {word_count}, char_count: {char_count}")#, text: {text[:200]}...")# , sentence: {sentence}")
        pass
    return truncated_text.strip()



def repeat_text(text, times):
    """textをtimes回繰り返す"""
    repeated_text = (text + " ") * times
    return repeated_text.strip()


# ***** 言語、国、都市のリストを取得する関数と、単語がそれらのリストに含まれるかを判断する関数 *****

def get_main_lang_lst(print_flag=False):
    """pycountryを使って、iso639-1の言語コードと対応する言語名のリストを取得する
    """
    iso639_1_languages = sorted(
        [
            {"code": lang.alpha_2, "name": lang.name}
            for lang in pycountry.languages
            if hasattr(lang, "alpha_2") and hasattr(lang, "name")
        ],
        key=lambda x: x["code"]
    ) # {'code': 'nl', 'name': 'Dutch'}, {'code': 'nn', 'name': 'Norwegian Nynorsk'},

    langs = [lang['name'] for lang in iso639_1_languages]

    if print_flag:
        print(f"Total languages: {len(iso639_1_languages)}, languages: {langs[:10]}, ...")
    return langs



def is_language(word, langs):
    """wordがlangs内の言語に該当するか、大文字小文字関係なく完全一致するかで判断する
    examples = [
        'English',
        'english',
        'This is English.', # no match
    ]
    """
    # langs内の言語に該当するか、大文字小文字関係なく完全一致するかで判断する正規表現パターンを作成
    lang_pattern = re.compile(r'\b(' + '|'.join(re.escape(lang) for lang in langs) + r')\b', re.IGNORECASE)

    # 完全一致で判断
    if lang_pattern.fullmatch(word.strip()):
        # print(f"'{word}' matches a language in the list.")
        return True
    else:
        # print(f"'{word}' does NOT match any language in the list.")
        return False
    

def get_country_lst(print_flag=False):
    """pycountryを使って、国名のリストを取得する
    """
    country_names = sorted({
        country.name
        for country in pycountry.countries
        if hasattr(country, "name") and country.name
    })
    if print_flag:
        print(f"Total countries: {len(country_names)}, examples: {country_names[:10]}, ...")

    # 冠詞theが必要な国名を追加
    countries_with_the = [
        "the Bahamas",
        "the Comoros",
        "the Czech Republic",
        "the Dominican Republic",
        "the Gambia",
        "the Maldives",
        "the Marshall Islands",
        "the Netherlands",
        "the Philippines",
        "the Solomon Islands",
        "the United Arab Emirates",
        "the United Kingdom",
        "the UK",
        "the United States",
        "the US",
        "the Central African Republic",
    ]
    return country_names + countries_with_the

def is_country(word, country_names):
    """wordがcountry_names内の国名に該当するか、大文字小文字関係なく完全一致するかで判断する
    examples = [
        'Japan',
        'japan',
        'This is Japan.', # no match
    ]
    """
    # country_names内の国名に該当するか、大文字小文字関係なく完全一致するかで判断する正規表現パターンを作成. 'the ' + countryも追加して、例えば "the United States" のような国名にも対応できるようにする
    # country_pattern = re.compile(r'\b(' + '|'.join(re.escape(country) for country in country_names) + r')\b', re.IGNORECASE)
    country_pattern = re.compile(r'\b(' + '|'.join(re.escape(country) for country in country_names + ['the ' + name for name in country_names]) + r')\b', re.IGNORECASE)

    # 完全一致で判断
    if country_pattern.fullmatch(word.strip()):
        # print(f"'{word}' matches a country name in the list.")
        return True
    else:
        # print(f"'{word}' does NOT match any country name in the list.")
        return False
    

def get_city_lst(print_flag=False):
    """geonamescacheを使って、人口40万人以上の都市のリストを取得する
    """
    gc = geonamescache.GeonamesCache(min_city_population=15000)

    major_cities = sorted(
        [
            {
                "name": city["name"],
                "countrycode": city["countrycode"],
                "population": int(city.get("population", 0)),
            }
            for city in gc.get_cities().values()
            if city.get("name") and int(city.get("population", 0)) >= 400_000
        ],
        key=lambda x: (-x["population"], x["name"])
    )
    city_names = [city["name"] for city in major_cities]
    if print_flag:
        print(f"Total major cities: {len(major_cities)}, examples: {city_names[:10]}, ...")

    return city_names

def is_city(word, city_names):
    """wordがcity_names内の都市名に該当するか、大文字小文字関係なく完全一致するかで判断する
    examples = [
        'Tokyo',
        'tokyo',
        'This is Tokyo.', # no match
    ]
    """
    # city_names内の都市名に該当するか、大文字小文字関係なく完全一致するかで判断する正規表現パターンを作成
    city_pattern = re.compile(r'\b(' + '|'.join(re.escape(city) for city in city_names) + r')\b', re.IGNORECASE)

    # 完全一致で判断
    if city_pattern.fullmatch(word.strip()):
        # print(f"'{word}' matches a city name in the list.")
        return True
    else:
        # print(f"'{word}' does NOT match any city name in the list.")
        return False



def get_year_if_it_is_year(word):
    """wordが4桁の数字で表される年であれば、その年を返す。そうでなければFalseを返す
    examples = [
        '1999', # returns '1999'
        '2020', # returns '2020'
        '1999 BC', # returns '1999 BC'
        'This is 1999.', # returns False
        '99', # returns False
    ]
    """
    # year_pattern = re.compile(r'^\d{4}(?:\s*BC)?$')  # 4桁の数字で、オプションで後ろに " BC" が続くパターン
    # 4桁までの数字 or 後ろに " BC" が続くパターン
    year_pattern = re.compile(r'^\d{1,4}(?:\s*BC)?$')
    if year_pattern.match(word.strip()):
        return word.strip()
    else:
        return False

def normalize_PublishedIn_facts(feats, template):
    """'PublishedIn' については、複数の出版年が全て列挙されている場合があるため、最初の年のみを使用する
    """
    new_publishedIn_feats = []  # 年以外の特徴はそのまま保持するためのリスト
    year_feats = []             # 年を表す特徴を保持するためのリスト
    mask_pattern = re.escape(template).replace("\\[MASK\\]", "(.+)") 
    for feat in feats:
        match = re.match(mask_pattern, feat)
        if not match:
            continue
        maskFeat = match.group(1)  # キャプチャグループから[MASK]部分を抽出
        # もしmaskFeatが年を表す4桁の数字であれば、year_featsに追加する
        if get_year_if_it_is_year(maskFeat):  # re.match(r'^\d{4}$', maskFeat):
            year_feats.append(maskFeat)
        else:
            # featが年以外であれば、そのままnew_publishedIn_featsに残す
            new_publishedIn_feats.append(feat)
    if year_feats != []:
        first_year = sorted(year_feats)[0] if year_feats else None
        new_publishedIn_feats.append(template.replace("[MASK]", first_year)) # 最初の年を[MASK]に入れた文をnew_publishedIn_featsに追加する
    return new_publishedIn_feats    # 年以外の特徴リスト + [一番最初の年]




# *** prompt構成のための関数 ***
def create_test_prompt(test_text, prompt_base, model_name):
    """test_textを埋め込み、modelに入力するためのpromptを作成する関数
    """
    if prompt_base is None or prompt_base.strip() == "":
        prompt = f"{test_text}\n\nAnswer: " # Answer:の後の空白は重要なのでstripしない. Qwenではこの空白を消すと, prob0.3以上の回答が179→29に減ったため, 空白が必要.
    else:
        prompt = f"{prompt_base.strip()}\n\n{test_text}\n\nAnswer: "
    if model_name.split("/")[-1].startswith("Qwen3") and not model_name.split("/")[-1].endswith("-Base"):
        # Qwen3-Instruct系の場合は思考モードをオフにする必要がある. https://arc.net/l/quote/uwefkzbz
        prompt = prompt + "/no_think"
    return prompt
