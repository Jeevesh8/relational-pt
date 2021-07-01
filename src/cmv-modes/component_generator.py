import bs4
import os
import re
from bs4 import BeautifulSoup
from typing import Optional


def clean_text(text):
    """Replaces HTML specific punctuations and symbols to their commonly occuring counterparts.
    Adds spaces around punctuations(excluding >, < symbols, which are used around xml/html tags).
    """
    #    print("Before replaces:", text)
    replaces = [("’", "'"), ("“", '"'), ("”", '"'), ("&", "and"),
                ("∆", "[DELTA]")]
    for elem in replaces:
        text = text.replace(*elem)

    for elem in [".", ",", "!", ";", ":", "*", "?", "/", '"', "(", ")", "^"]:
        text = text.replace(elem, " " + elem + " ")

    #    print("After replaces:", text)
    return text


def add_tags(post, user_dict):
    """Adds user, url, and quote tags. Adds spaces around <claim>, </claim>, <premise>, </premise> tags.
    Additionally tries to remove away some footnotes.
    Args:
        post:       The text of a post, having claim, premise tags as occuring in the original xml file.
        user_dict:  A dict of already existing users for the current thread. Maintain one dict per thread.
    Returns:
        The post with the modifications and the user tag(str of for [USERi]).
    """
    if post["author"] not in user_dict:
        user_dict[post["author"]] = len(user_dict)

    text = str(post)  # Note 2
    #    print("Before adding tags:", text)

    user_tag = "[USER" + str(user_dict[post["author"]]) + "]"

    pattern0 = r"(\n\&gt;\*Hello[\S]*)"
    pattern1 = r"(https?://)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*(\.html|\.htm)*"
    pattern2 = r"\&gt;(.*)\n"

    text = text.replace("</claim>", "</claim> ")
    text = text.replace("<claim", " <claim")
    text = text.replace("<premise", " <premise")
    text = text.replace("</premise>", "</premise> ")
    text = re.sub(pattern0, "", text)  # Replace Footnotes
    text = re.sub(pattern1, "[URL]", text)  # Replace [URL]
    text = re.sub(pattern2, "[STARTQ]" + r"\1" + "[ENDQ] ",
                  text)  # Replace quoted text
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    #    print("after adding tags:", text)
    return str(text), user_tag


def get_components(
    component: bs4.BeautifulSoup,
    parent_type: Optional[str] = None,
    parent_id: Optional[str] = None,
    parent_refers: Optional[str] = None,
    parent_rel_type: Optional[str] = None,
):
    """Yields nested components from a parsed component one-by-one. In the form:
    (text, type, id, refers, relation_type)
    text:          The text of the component
    type:          other/claim/premise
    id:            The id of the component, None for Non-Argumentative
    refers:        The ids of the component the current component is related to. None, if it isn't related to any component(separated with _)
    relation_type: The type of relation between current component and refers. None, iff refers is None.
    """
    def chain_yield(comp_type="claim"):
        nonlocal component
        component = str(component)
        parsed_component = BeautifulSoup(component, "xml")

        parent_id = str(parsed_component.find(comp_type)["id"])
        try:
            parent_refers = str(parsed_component.find(comp_type)["ref"])
            parent_rel_type = str(parsed_component.find(comp_type)["rel"])
        except:
            parent_refers = None
            parent_rel_type = None

        for part in parsed_component.find(comp_type).contents:

            if (not str(part).strip().startswith("<claim")
                    and not str(part).strip().startswith("<premise") and
                    not part == parsed_component.find(comp_type).contents[0]):
                parent_ref = parent_id
                parent_id += "Ć"
                parent_rel_type = "cont"

            for _ in get_components(part, comp_type, parent_id, parent_refers,
                                    parent_rel_type):
                yield _

    if str(component).strip() == "":
        yield None

    elif str(component).strip().startswith("<claim"):
        for _ in chain_yield(comp_type="claim"):
            yield _

    elif str(component).strip().startswith("<premise"):
        for _ in chain_yield(comp_type="premise"):
            yield _

    else:
        if clean_text(str(component).strip()) == "":
            print("Component reduced to empty by cleaning: ", str(component))
        yield (
            clean_text(str(component).strip()),
            "other" if parent_type is None else parent_type,
            parent_id,
            parent_refers,
            parent_rel_type,
        )


def generate_components(filename):
    """Yields components from a thread one-by-one. In the form:
    (text, type, id, refers, relation_type)
    text:          The text of the component
    type:          other/claim/premise
    id:            The id of the component, None for Non-Argumentative
    refers:        The ids of the component the current component is related to. None, if it isn't related to any component(separated with _)
    relation_type: The type of relation between current component and refers. None, iff refers is None.
    """

    with open(filename, "r") as g:
        xml_str = g.read().replace("&#8217", " &#8217 ")

    xml_with_html_substituted = str(BeautifulSoup(xml_str, "lxml"))
    parsed_xml = BeautifulSoup(xml_with_html_substituted, "xml")  # Note 1.

    if len(re.findall(r"\&\#.*;", str(parsed_xml))) != 0:
        raise AssertionError("HTML characters still remaining in XML: " +
                             str(re.findall(r"\&\#.*;", str(parsed_xml))))

    user_dict = dict()

    yield (parsed_xml.find("title").get_text(), "claim", "title", None, None)

    for post in [parsed_xml.find("op")] + parsed_xml.find_all("reply"):
        modified_post, user_tag = add_tags(post, user_dict)
        parsed_modified_post = BeautifulSoup(modified_post, "xml")

        try:
            contents = parsed_modified_post.find("reply").contents
        except:
            contents = parsed_modified_post.find("op").contents

        yield (user_tag, "user_tag", None, None, None)

        for component in contents:
            for elem in get_components(component):
                if elem is not None:
                    yield elem


def get_all_threads():
    for t in ["negative", "positive"]:
        root = "AmpersandData/change-my-view-modes/v2.0/" + t + "/"
        for f in os.listdir(root):
            filename = os.path.join(root, f)

            if not (os.path.isfile(filename) and f.endswith(".xml")):
                continue

            for elem in generate_components(filename):
                yield elem


"""NOTES:
1. Deltas(&#8710) and some other symbols not parsed correctly without this double parsing. &gt; (>) , &#8217(apostrophe) get parsed fine with just using 'xml' and lxml.
2. &gt; doesn't get parsed to > when using str(post). But when doing post.get_text(), all the tags inside post will be removed and &gt; will be parsed to ">". 
   The &# characters in initial xml_string are all converted to proper unicode versions, the moment we parse with "lxml".
"""
