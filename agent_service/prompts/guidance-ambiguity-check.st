用户问题：{question}

以下是意向分类的候选品类及其匹配分数：
{candidates}

请判断：
1. 用户的问题是否存在品类歧义，即无法确定用户想问哪个品类？
2. 如果存在歧义，输出 true；如果用户已经明确品类，输出 false。

只输出 JSON：
{"ambiguous": true, "category_ids": [], "reason": "判断理由"}
