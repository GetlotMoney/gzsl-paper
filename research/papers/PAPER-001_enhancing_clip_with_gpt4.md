# PAPER-001：Enhancing CLIP with GPT-4

```yaml
paper_id: PAPER-001
title: "Enhancing CLIP with GPT-4: Harnessing Visual Descriptions as Prompts"
authors: Mayug Maniparambil; Chris Vorster; Derek Molloy; Noel Murphy; Kevin McGuinness; Noel E. O'Connor
year: 2023
venue: ICCV Workshops
arxiv: "2307.11661"
publisher_url: "https://openaccess.thecvf.com/content/ICCV2023W/MMFM/html/Maniparambil_Enhancing_CLIP_with_GPT-4_Harnessing_Visual_Descriptions_as_Prompts_ICCVW_2023_paper.html"
source_checked_at: 2026-08-22
pdf_uri: "C:\\Users\\Administrator\\Desktop\\CV论文\\2023\\Enhancing CLIP with GPT-4：Harnessing Visual Descriptions as Prompts.pdf"
pdf_sha256: 31e006252a53f4646e3900cb4e06d6139ce80605466477c7642a30d97ae0577f
```

## 与本项目关系

论文使用GPT-4生成多句类别视觉描述并进行聚合。TG-VPR-H1是在该思路上的改版：把自由句子聚合改为local/unique/overall三组结构，并加入共享Value重参数化、固定等权和topology约束。这些改动属于本项目，不是论文原结论。
