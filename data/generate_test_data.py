"""Script to generate test datasets (3,000 and 300 entries) for game localization."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path


def generate_game_strings(target_count: int = 3000, seed: int = 42) -> list[str]:
    """Generates a list of unique Japanese game localization text strings."""
    rng = random.Random(seed)
    entries: set[str] = set()

    def add(item: str) -> bool:
        item = item.strip()
        if not item or item in entries:
            return False
        entries.add(item)
        return True

    # Ratio allocations based on target_count
    if target_count <= 500:
        n_ui = max(20, int(target_count * 0.12))
        n_sys = max(30, int(target_count * 0.16))
        n_items = max(30, int(target_count * 0.18))
        n_skills = max(15, int(target_count * 0.08))
        n_dialogue = max(60, int(target_count * 0.32))
        n_escape = max(15, int(target_count * 0.06))
        n_lore = max(10, int(target_count * 0.04))
        n_dev = max(10, int(target_count * 0.04))
    else:
        n_ui = 250
        n_sys = 350
        n_items = 600
        n_skills = 250
        n_dialogue = 1000
        n_escape = 150
        n_lore = 250
        n_dev = 150

    # 1. UI & Menu Labels
    ui_bases = [
        "ニューゲーム", "コンティニュー", "セーブ", "ロード", "オプション",
        "アイテム", "スキル", "装備", "ステータス", "クエスト", "図鑑", "隊列",
        "ショップ", "実績", "ヘルプ", "タイトルへ", "ゲーム終了", "決定", "キャンセル",
        "全員回復", "ソート", "まとめて売却", "武器", "防具", "装飾品", "貴重品",
        "消費アイテム", "素材", "キーアイテム", "大事なもの", "魔法", "奥義",
        "パッシブ", "アビリティ", "作戦", "戦闘設定", "音量設定", "画面設定",
        "言語設定", "操作説明", "クレジット", "オート送り", "早送り", "バックログ",
        "メッセージ速度", "ウィンドウカラー", "ダッシュ設定", "難易度設定",
        "イージー", "ノーマル", "ハード", "ナイトメア", "マスター音量", "BGM音量",
        "SE音量", "ボイス音量", "全画面表示", "ウィンドウ表示", "解像度",
        "垂直同期", "アンチエイリアス", "明暗度調整", "テキスト速度",
        "自動戦闘", "戦闘アニメ", "逃げる", "防御", "通常攻撃", "交代",
        "はい", "いいえ", "購入", "売却", "所持数", "必要素材", "鍛冶屋",
        "酒場", "宿屋", "教会", "ギルド", "闘技場", "転送陣", "案内所",
        "倉庫", "保管庫", "調合", "錬金術", "分解", "強化", "覚醒",
        "リセット", "デフォルト", "適用", "閉じる", "戻る", "次へ", "前へ"
    ]
    for u in ui_bases[:n_ui]:
        add(u)

    # 2. System Messages & Battle Prompts
    system_templates = [
        "セーブデータを保存しました。",
        "セーブスロット {slot} に記録しました。",
        "データを上書きしますか？",
        "本当にゲームを終了しますか？",
        "未保存の進行状況は破棄されます。",
        "所持金が足りません。",
        "所持金が {gold} G 不足しています。",
        "これ以上アイテムを持てません。",
        "インベントリの空きがありません。",
        "このアイテムは装備できません。",
        "現在のクラスでは習得できません。",
        "レベルが不足しています。（必要Lv.{lvl}）",
        "装備を変更しました。",
        "HPが全快した！",
        "HPとMPが完全に回復した！",
        "状態異常が治癒した。",
        "戦闘不能から回復した！",
        "会心の一撃！！ {dmg} のダメージを与えた！",
        "痛恨の一撃！！ {dmg} のダメージを受けた！",
        "ミス！攻撃は当たらなかった。",
        "攻撃は回避された！",
        "効果がなかった……",
        "レベルが上がった！",
        "Lv.{lvl} に到達した！",
        "攻撃力が {stat} ポイント上昇した！",
        "防御力が {stat} ポイント上昇した！",
        "魔力が {stat} ポイント上昇した！",
        "素早さが {stat} ポイント上昇した！",
        "新しいスキル【{skill}】を習得した！",
        "クエスト【{quest}】を受注しました。",
        "クエスト【{quest}】を達成しました！",
        "クエスト【{quest}】が更新されました。",
        "報酬として {gold} G を獲得した！",
        "経験値 {exp} EXP を獲得した！",
        "アイテム【{item}】を手に入れた！",
        "{item} を {num} 個失った。",
        "扉には頑丈な鍵がかかっている。",
        "【{item}】を使って扉を開けた。",
        "ここからは進めないようだ。",
        "どこかで何かが作動する音がした。",
        "罠が発動した！針が飛び出してきた！",
        "宝箱を開けた！",
        "しかし宝箱はミミックだった！",
        "全滅してしまった……",
        "戦闘から逃走した。",
        "しかし逃げ切れなかった！",
    ]
    slots = [1, 2, 3, 5]
    golds = [100, 500, 1000, 5000]
    lvls = [10, 20, 30, 50, 99]
    stats = [2, 5, 10]
    exps = [100, 500, 2500]
    nums = [1, 2, 5]
    sample_skills = ["旋風脚", "紅蓮炎舞", "光のヴェール", "瞬歩", "聖なる防壁", "暗黒剣"]
    sample_quests = ["ゴブリンの脅威", "薬草の採取", "古城の調査", "盗賊団の討伐", "竜の目覚め"]
    sample_items = ["古びた銅の鍵", "銀の鍵", "黄金の鍵", "封印の結晶", "神秘のペンダント"]

    count_sys = 0
    for tmpl in system_templates:
        if count_sys >= n_sys:
            break
        rendered = tmpl.format(
            slot=rng.choice(slots),
            gold=rng.choice(golds),
            lvl=rng.choice(lvls),
            stat=rng.choice(stats),
            exp=rng.choice(exps),
            num=rng.choice(nums),
            dmg=rng.randint(50, 9999),
            skill=rng.choice(sample_skills),
            quest=rng.choice(sample_quests),
            item=rng.choice(sample_items),
        )
        if add(rendered):
            count_sys += 1

    # 3. Protected Katakana & Fantasy Suffix Items
    katakana_prefixes = ["ハイ", "メガ", "エリ", "グランド", "ミスリル", "ドラゴン", "ホーリー", "ダーク"]
    katakana_bases = ["ポーション", "エーテル", "エリクサー", "ソード", "シールド", "アーマー", "リング", "ロッド"]
    count_items = 0
    for p, b in itertools.product(katakana_prefixes, katakana_bases):
        if count_items >= n_items // 2:
            break
        if add(f"{p}{b}"):
            count_items += 1

    fantasy_elements = ["竜", "生命", "世界樹", "神聖", "賢者", "紅蓮", "漆黒", "氷雪", "雷光", "精霊", "光輝", "封印"]
    fantasy_suffixes = ["の花", "の草", "の薬", "の種", "の果実", "の石", "の剣", "の盾", "の鎧", "の指輪", "の巻物", "の鍵", "の羽"]
    for e, s in itertools.product(fantasy_elements, fantasy_suffixes):
        if count_items >= n_items:
            break
        if add(f"{e}{s}"):
            count_items += 1

    # 4. Statuses & Skills
    statuses = [
        "毒", "猛毒", "麻痺", "睡眠", "沈黙", "暗闇", "混乱", "魅了", "石化", "即死",
        "呪い", "衰弱", "火傷", "凍結", "感電", "リジェネ", "ヘイスト", "プロテス"
    ]
    count_skills = 0
    for s in statuses:
        if count_skills >= n_skills // 2:
            break
        if add(s):
            count_skills += 1
        if add(f"{s}状態になった！"):
            count_skills += 1

    skill_cores = [
        "一閃", "乱舞", "次元斬", "鳳凰拳", "メテオ", "インフェルノ",
        "サンダーボルト", "ホーリーレイ", "シャドウエッジ", "旋風刃"
    ]
    for c in skill_cores:
        if count_skills >= n_skills:
            break
        if add(c):
            count_skills += 1
        if add(f"真・{c}"):
            count_skills += 1

    # 5. Dialogue & Character Quotes
    characters = ["勇者", "魔王", "アリス", "エリス", "ルシアン", "シオン", "カレン", "賢者", "村長", "騎士団長"]
    dialogue_phrases = [
        "ここから先へは行かせないわ！",
        "我が刃の錆にしてくれる！覚悟しろ！",
        "仲間たちのためにも、絶対に負けられないんだ！",
        "覚悟を決める時が来たようだな。",
        "この古の封印が解かれるとき、世界は深い闇に包まれるだろう……",
        "禁忌の書に記されし詠唱を始めます。力を貸してください。",
        "光の導きが、我らの行く手に幸運をもたらさんことを。",
        "いらっしゃい！今夜はゆっくり休んでいきな。",
        "珍しいお宝を持ってきたぜ。見ていくかい？",
        "どうして……どうしてこんなことになってしまったの？",
        "すまない、私の力不足だった……",
        "諦めてはいけない！まだ最後の希望は残っている！",
        "頼んだぞ、若き旅人よ。この国の未来はそなたに託された。",
        "待って！その道を行くのは危険すぎるわ！",
        "あの扉の奥に、伝説の秘宝が眠っているという噂だ。",
        "信じているよ。君ならきっと、困難を乗り越えられるはずだ。",
        "何があっても、最後まであなたと一緒に行きます！",
        "おいおい、冗談だろ？あんな化物と戦う気か！？",
        "ふん、所詮はその程度の力か。期待外れだな。",
        "さあ、宴の始まりだ！思い切り暴れてやろうぜ！",
        "お願い……もう争うのはやめて！誰も傷ついてほしくないの！",
        "これを受け取ってください。きっとあなたの旅の助けになるはずです。",
        "血の匂いが立ち込めている。警戒を怠るな！",
        "ククク……愚かな人間どもめ、恐怖に震えるがよい！",
        "見つけたぞ！探していた手掛かりはこれだ！",
        "約束は守るぜ。これが依頼の報酬だ、受け取りな。"
    ]
    count_dialogue = 0
    for char in characters:
        for phrase in dialogue_phrases:
            if count_dialogue >= n_dialogue:
                break
            r = rng.random()
            line = f"【{char}】\n「{phrase}」" if r < 0.4 else (f"【{char}】「{phrase}」" if r < 0.7 else f"「{phrase}」")
            if add(line):
                count_dialogue += 1

    # 6. Escape Code Sequences
    escape_templates = [
        r"\N[1]「任せておいてくれ！必ず成し遂げてみせる！」",
        r"\N[2]「\C[2]危険地帯\C[0]に一人で行く気？無茶よ！」",
        r"\N[1]「これを受け取ってくれ。\I[15] \C[3]聖なるペンダント\C[0]だ。」",
        r"\N[3]「\V[10] ゴールドで買い取ろう。どうかな？」",
        r"\!「な、なんだって……！？\.そんな馬鹿な！」",
        r"\^「……消えた！？一体どこへ行ったんだ！」",
        r"所持金が \C[14]\G\C[0] 増えた！",
        r"\C[1]{char}\C[0]の好感度が \C[2]+{val}\C[0] 上昇した！",
        r"\I[{icon}] 【\C[4]{item}\C[0]】を手に入れた！",
        r"\N[1]は \C[6]{skill}\C[0] の奥義を覚醒させた！",
        r"\V[{var}] の大ダメージを受けた！危ない！",
        r"\C[10]※警告※\C[0] この先は強敵が出現します。セーブを推奨します。"
    ]
    count_escape = 0
    for _ in range(n_escape * 2):
        if count_escape >= n_escape:
            break
        t = rng.choice(escape_templates)
        rendered = t.format(
            char=rng.choice(characters),
            item=rng.choice(sample_items),
            skill=rng.choice(sample_skills),
            val=rng.choice([1, 5, 10]),
            icon=rng.randint(1, 100),
            var=rng.randint(1, 20),
        )
        if add(rendered):
            count_escape += 1

    # 7. Lore & Dungeons
    lore_list = [
        "始まりの平原・最下層", "静寂の迷宮・地下1階", "試練の古城・謁見の間",
        "灼熱の火山洞窟・祭壇", "凍てつく雪原・地下深層", "星降る丘・一層",
        "遠い昔、神と魔が大地を揺るがして争った記録が記されている。",
        "壁面には風化した古代のレリーフが刻まれており、歴史の重みを感じさせる。",
        "祭壇の中央に青白く輝く水晶が鎮座し、神秘的な光彩を放っている。",
        "冷たい風が吹き抜け、背筋に寒気が走った。",
        "苔むした石畳の隙間から、名も知らぬ小さな花が健気に咲いている。",
        "かすかな水滴の落ちる音だけが、静まり返った洞窟内に反響している。"
    ]
    for l in lore_list[:n_lore]:
        add(l)

    # 8. Dev & Engine Quarantine Items
    dev_items = [
        "// TODO: イベントスキップ時のフラグ整合性を確認すること",
        "// FIXME: ボス第2形態移行時のBGMクロスフェード修正",
        "// NOTE: ここでプレイヤーの移動速度を一時的に半減させる",
        "/* 一時的にコメントアウト。製品版で有効化 */",
        "# スクリプト実行用エントリーポイント",
        "<!-- UIレイアウトの暫定アンカー配置 -->",
        "【開発】このマップのSE割り当ては次回ミーティングで確定",
        "【仕様】所持金の上限値は9,999,999に設定",
        "【デバッグ】デバッグ用全アイテム所持スイッチ",
        "【テスト】戦闘テスト用NPC会話フラグ",
        "TODO: 英語ボイスのリップシンクタイミング調整",
        "DEBUG: レベル即座に99へブースト",
        "bgm_battle_boss01.ogg",
        "se_sword_slash_heavy.wav",
        "voice_hero_attack01.ogg",
        "icon_potion_red.png",
        "event01【10】",
        "フレーム 60",
        "scene_intro_start",
        "SYS_AUDIO_MASTER_VOL",
        "================================",
        "▼▼▼ 次のページへ ▼▼▼"
    ]
    for d in dev_items[:n_dev]:
        add(d)

    # If count < target_count, procedural fill
    counter = 1
    while len(entries) < target_count:
        add(f"【記録断片 No.{counter:04d}】失われた古代王国の年代記の一節。")
        counter += 1

    result_list = sorted(list(entries))[:target_count]
    return result_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate game localization test JSON.")
    parser.add_argument("--count", type=int, default=300, help="Number of entries to generate")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    args = parser.parse_args()

    count = args.count
    output_path = (
        Path(args.output) if args.output else Path(f"E:/ai/projects/mtool_json_translate_local_lm/data/raw/test_{count}.json")
    )
    items = generate_game_strings(target_count=count)
    data_dict = {item: item for item in items}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(data_dict)} unique entries and saved to {output_path}")


if __name__ == "__main__":
    main()
