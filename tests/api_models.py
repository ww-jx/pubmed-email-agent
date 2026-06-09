from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict

# Fetch (XML -> Dict)


class DescriptorName(BaseModel):
    model_config = ConfigDict(extra="ignore")
    ui: str = Field(alias="@UI", default="")
    major_topic: str = Field(alias="@MajorTopicYN", default="N")
    text: str = Field(alias="#text", default="")


class MeshHeading(BaseModel):
    model_config = ConfigDict(extra="ignore")
    descriptor_name: DescriptorName = Field(alias="DescriptorName")
    qualifier_name: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        alias="QualifierName", default=None
    )


class MeshHeadingList(BaseModel):
    model_config = ConfigDict(extra="ignore")
    mesh_heading: Union[MeshHeading, List[MeshHeading]] = Field(alias="MeshHeading")


class Keyword(BaseModel):
    model_config = ConfigDict(extra="ignore")
    major_topic: Optional[str] = Field(alias="@MajorTopicYN", default=None)
    text: str = Field(alias="#text", default="")


class KeywordList(BaseModel):
    model_config = ConfigDict(extra="ignore")
    keyword: Union[Keyword, List[Keyword]] = Field(alias="Keyword")
    owner: Optional[str] = Field(alias="@Owner", default=None)


class Article(BaseModel):
    model_config = ConfigDict(extra="ignore")
    article_title: Union[str, Dict[str, Any]] = Field(alias="ArticleTitle")


class MedlineCitation(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pmid: Dict[str, str] = Field(alias="PMID")
    article: Article = Field(alias="Article")
    mesh_heading_list: Optional[MeshHeadingList] = Field(
        alias="MeshHeadingList", default=None
    )
    keyword_list: Optional[KeywordList] = Field(alias="KeywordList", default=None)


class FetchArticleModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    medline_citation: MedlineCitation = Field(alias="MedlineCitation")
    pubmed_data: Optional[Dict[str, Any]] = Field(alias="PubmedData", default=None)


# ESearch (JSON)


class ESearchResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    count: Union[str, int]
    retmax: Union[str, int]
    retstart: Union[str, int]
    idlist: List[str]


class ESearchResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    esearchresult: ESearchResult


# ELink (JSON)


class Link(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    score: Optional[str] = None


class LinkSetDb(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dbto: str
    linkname: str
    links: Optional[List[Link]] = None


class LinkSet(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dbfrom: str
    ids: List[str]
    linksetdbs: List[LinkSetDb]


class ELinkResponseModel(BaseModel):
    model_config = ConfigDict(extra="ignore")
    linksets: List[LinkSet]
