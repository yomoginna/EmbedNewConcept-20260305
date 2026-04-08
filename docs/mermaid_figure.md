```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#e1f5fe', 'edgeLabelBackground':'#ffffff', 'tertiaryColor': '#fff'}}}%%
graph TD
    %% スタイルの定義
    classDef service fill:#fff3e0,stroke:#ff9800,stroke-width:2px,rx:10,ry:10;
    classDef data fill:#e3f2fd,stroke:#2196f3,stroke-width:1px,rx:5,ry:5;
    classDef process fill:#e8f5e9,stroke:#4caf50,stroke-width:1px;
    classDef manual fill:#fce4ec,stroke:#e91e63,stroke-width:1px,stroke-dasharray: 5 5;
    classDef highlight fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;

    %% --- 外部サービス ---
    subgraph Services [外部データソース]
        DBpedia[DBpedia<br/>(知識ベース)]:::service
        Wikipedia[Wikipedia<br/>(百科事典)]:::service
        Wikidata[Wikidata<br/>(知識ベース)]:::service
    end

    %% --- フェーズ①：カテゴリの定義 ---
    subgraph Phase1 [① カテゴリの選定]
        direction TB
        GetTop[大カテゴリのリストアップ]:::process
        RawTopList[大カテゴリリスト<br/>(URI, ラベル)]:::data
        
        Curate[<b>手動選別</b><br/>実体を持つクラスに絞り込み]:::manual
        CuratedTopList[精査済み大カテゴリリスト]:::data
        
        GetMid[中カテゴリのリストアップ<br/>(子クラス、再帰取得)]:::process
        MidCatMap[大-中カテゴリ対応表]:::data
        MidList[中カテゴリリスト<br/>(URI, ラベル)]:::data
    end

    %% --- フェーズ②：固有名詞の抽出 ---
    subgraph Phase2 [② 固有名詞の抽出]
        direction TB
        GetNouns[カテゴリに属する<br/>固有名詞の抽出]:::process
        NounLists[カテゴリ別固有名詞リスト<br/>(QID, ラベル等)]:::data
    end

    %% --- フェーズ③：特徴文と学習データの生成 ---
    subgraph Phase3 [③ 学習データの生成]
        direction TB
        GenFeatures[<b>生成AIによる特徴文生成</b><br/>(Gemini, プロンプト)]:::process
        FactJSONs[概念別 特徴文データ<br/>(JSON)]:::highlight
        
        Construct[学習用データの構築]:::process
        TrainData[最終的な学習用データセット]:::data
    end

    %% --- データの流れ（矢印） ---
    
    %% フェーズ1の流れ
    DBpedia --> GetTop
    GetTop --> RawTopList
    RawTopList --> Curate
    Curate --> CuratedTopList
    CuratedTopList --> GetMid
    DBpedia -.-> GetMid
    GetMid --> MidCatMap
    GetMid --> MidList

    %% フェーズ2の流れ
    MidList --> GetNouns
    Wikidata --> GetNouns
    GetNouns --> NounLists

    %% フェーズ3の流れ
    NounLists --> GenFeatures
    Wikipedia --> GenFeatures
    GenFeatures --> FactJSONs
    FactJSONs --> Construct
    Construct --> TrainData
```