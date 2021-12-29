import re
import json
import numpy as np
import pandas as pd


def compare(df1, df2):
    """Compare two DataFrame and return the difference"""
    df = pd.concat([df1, df2])
    df = df.drop_duplicates(keep=False).reset_index(drop=True)
    return df


def compare_and_add(df1, df2):
    """Compare two DataFrame and return the difference and add one column with state"""
    df_dict = dict(df1=df1, df2=df2)
    df = pd.concat(df_dict)
    df = df.drop_duplicates(keep=False)
    column_state = []
    print
    for belong, index in df.index:
        if belong == "df1":
            column_state.append("Removed")
        else:
            column_state.append("Added")
    df["State"] = column_state
    df = df.reset_index(drop=True)
    return df


def strip(df):
    df.columns = df.columns.str.strip()
    df_obj = df.select_dtypes(["object"])
    df[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    return df


def write_md(fname, df):
    with open(fname, "w", encoding="utf-8") as f:
        df.to_markdown(f)


def read_md(fname):

    # read markdown files
    try:
        df = pd.read_table(fname, sep="|", header=0, index_col=1, skipinitialspace=True)
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("No data.")
        return None
    # drop the first and the last columns
    df = df.iloc[1:].drop(df.columns[[0, -1]], axis=1)
    # strip strings
    df = strip(df)

    # replace `NaN` with `None`
    df = df.where(pd.notnull(df), None)
    # replace datatypes
    df = df.astype({"Year": "int64"})
    return df


def parse_paper(line):
    instance = {}
    content = line.split("*, ")
    title = venue = code = None
    if len(content) == 3:
        title, venue, code = content
    else:
        title, venue = content

    end = venue.find("]")
    year = re.findall(r"\d+", venue[max(end - 5, 0): end])

    if year:
        year = int(year[0])
        if year < 2000:
            year += 2000

    instance["Title"] = title
    instance["Venue"] = venue
    instance["Year"] = year
    instance["Code"] = code

    return instance


if __name__ == '__main__':

    print("#" * 10, "Begin", "#" * 10)

    with open("README.md", "r", encoding="UTF-8") as f:
        content = f.readlines()

    # parsing
    i = 0
    papers = []
    titles = []

    while i < len(content):
        line = content[i]
        if line.strip() == "# 🔗Resource":
            break

        if line.startswith("# "):
            this_level = 0
            titles = []
            titles.append(line[this_level + 2:].strip())
        elif line.startswith("## "):
            this_level = 1
            while len(titles) > this_level:
                titles.pop()
            titles.append(line[this_level + 2:].strip())

        elif line.startswith("### "):
            this_level = 2
            while len(titles) > this_level:
                titles.pop()
            titles.append(line[this_level + 2:].strip())

        elif line.startswith("#### "):
            this_level = 3
            while len(titles) > this_level:
                titles.pop()
            titles.append(line[this_level + 2:].strip())

        is_paper = line.startswith("+ ")

        if is_paper:
            line = line[2:].strip()
            paper = parse_paper(line)
            paper["belong"] = titles[:]
            if not paper["Year"] and len(titles) > 1:
                paper["Year"] = int(titles[-1])
            paper["Type"] = titles[0].strip()
            papers.append(paper)
        i += 1

    ################################### Step1: To pandas DataFrame ######################################################################
    tb = []
    columns = ["Title", "Type", "Venue", "Code", "Year"]
    for paper in papers:
        tmp = []
        for col in columns:
            if isinstance(paper[col], str):
                tmp.append(paper[col].strip("*"))
            else:
                tmp.append(paper[col])

        tb.append(tmp)
    tb = pd.DataFrame(tb, columns=columns)
    # strip
    tb = strip(tb)

    ################################### Step2: Find papers with code ######################################################################
    tb_with_code = tb.drop_duplicates(subset=["Title"])
    tb_with_code = tb_with_code[pd.notna(tb_with_code["Code"])].reset_index(drop=True)
    write_md("Categorized/papers_with_code.md", tb_with_code)

    ################################### Step3: Categorize papers by Title ######################################################################
    # Sorted by Title
    tb_by_title = (
        tb.drop_duplicates(subset=["Title"]).sort_values("Title").reset_index(drop=True)
    )
    write_md("Categorized/alphabet.md", tb_by_title)

    ################################### Step4: Categorize papers by Year ######################################################################
    tb_by_year = (
        tb.drop_duplicates(subset=["Title"]).sort_values("Type").reset_index(drop=True)
    )
    tb_by_year = tb_by_year.sort_values(
        "Year", kind="mergesort", ascending=False
    ).reset_index(drop=True)

    # read before write
    tb_before = read_md("Categorized/year.md")
    write_md("Categorized/year.md", tb_by_year)

    ################################### Step5: find recently updated and outdated papers ######################################################################
    tb_now = tb_by_year
    # find recently updadted papers
    diff = compare_and_add(tb_before, tb_now)

    # check if recently updadted papers are outdated in 30 days
    now = pd.Timestamp.today()
    # check outdated if 30 days passed
    outdate = now - pd.Timedelta(30, unit="D")
    recent_tb = read_md("Categorized/recent.md")
    if recent_tb is not None and len(recent_tb):
        recent_tb["Date"] = pd.to_datetime(recent_tb["Date"])
        recent_tb = recent_tb[recent_tb["Date"] > outdate]
    # if there is some newly added papers
    if len(diff):
        arr = diff.to_numpy()
        arr_out = []
        for line in arr:
            title, types, venue, code, year, state = line
            arr_out.append(np.array([title, types, venue, code, year, state, now]))
        df_new = pd.DataFrame(np.array(arr_out), columns=diff.columns.to_list() + ["Date"])
        if recent_tb is not None and len(recent_tb):
            recent_tb = df_new.append(recent_tb)
        else:
            recent_tb = df_new
        recent_tb.drop_duplicates(subset=["Title"]).reset_index(drop=True)
        recent_tb = recent_tb.sort_values(
            "Date", kind="mergesort", ascending=False
        ).reset_index(drop=True)

    # write back
    if recent_tb is not None and len(recent_tb):
        recent_tb["Date"] = recent_tb["Date"].dt.strftime("%Y-%m-%d")
        write_md("Categorized/recent.md", recent_tb)

    ################################### Step6: Categorize papers by venue ######################################################################
    arr = tb.to_numpy()
    confs = [
        "AAAI",
        "IJCAI",
        "ICLR",
        "WWW",
        "KDD",
        "ICML",
        "TKDE",
        "CIKM",
        "WSDM",
        "NeurIPS",
        "USENIX",
        "ICDM",
        "Arxiv",
        "UAI",
        "ICSE",
        "ECAI",
        "Others",
    ]
    arr_out = []
    for line in arr:
        title, types, venue, code, year = line
        pubs = None
        for c in confs:
            if c == "KDD" and c in venue and not ("PAKDD" in venue and "PKDD" in venue):
                pubs = c
                break
            elif c in venue:
                pubs = c
                break
        pubs = pubs or "Others"
        arr_out.append(np.array([title, types, venue, code, year, pubs]))

    tb_by_venue = pd.DataFrame(np.array(arr_out), columns=columns + ["Pubs"])
    tb_by_venue = tb_by_venue.sort_values("Pubs", kind="mergesort").reset_index(drop=True)

    for i, c in enumerate(confs):
        t = tb_by_venue[tb_by_venue["Pubs"] == c].reset_index(drop=True)
        t = t.drop('Pubs', axis=1)
        mod = "w" if i == 0 else "a"
        with open("Categorized/venue.md", mod, encoding="utf-8") as f:
            f.writelines("# " + c + "\n")
            t.to_markdown(f)
            f.writelines("\n")

    # ################################### Step7: Write README.md ######################################################################
    # for i in range(100):
    #     if "(Updating " in content[i]:
    #         begin = content[i].index("(Updating ")
    #         end = content[i].index(")")
    #         content[i] = (
    #             content[i][:begin] + f"(Updating {len(tb)} papers)" + content[i][end + 1:]
    #         )
    # with open("README.md", "w", encoding="utf-8") as f:
    #     for line in content:
    #         f.writelines(line)
    print(f"{len(tb)} papers in total.")
    print("#" * 10, "End", "#" * 10)
