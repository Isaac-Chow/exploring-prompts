import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional
from models import PromptTemplate


class PromptLoader:
    """Loads and manages XML prompt templates."""
    
    def __init__(self, prompts_dir: Optional[str] = None):
        if prompts_dir is None:
            prompts_dir = Path(__file__).parent / "prompts"
        self.prompts_dir = Path(prompts_dir)
        self._cache: dict[str, PromptTemplate] = {}
        
    def list_prompts(self) -> list[str]:
        """List all available prompt template names."""
        prompts = []
        for file in self.prompts_dir.glob("prompt-*.xml"):
            name = file.stem.replace("prompt-", "")
            prompts.append(name)
        return sorted(prompts)
    
    def load_prompt(self, name: str) -> PromptTemplate:
        """Load a prompt template by name."""
        if name in self._cache:
            return self._cache[name]
        
        file_path = self.prompts_dir / f"prompt-{name}.xml"
        if not file_path.exists():
            raise FileNotFoundError(f"Prompt template not found: {name}")
        
        template = self._parse_xml(file_path)
        self._cache[name] = template
        return template
    
    def _parse_xml(self, file_path: Path) -> PromptTemplate:
        """Parse an XML prompt template file."""
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        name = root.get('name', file_path.stem)
        version = root.get('version', '1.0')
        
        system_elem = root.find('system_prompt')
        user_elem = root.find('user_template')
        
        if system_elem is None or user_elem is None:
            raise ValueError("Invalid prompt template: missing system_prompt or user_template")
        
        system_prompt = self._extract_text(system_elem)
        user_template = self._extract_text(user_elem)
        
        return PromptTemplate(
            name=name,
            version=version,
            system_prompt=system_prompt,
            user_template=user_template,
            file_path=str(file_path)
        )
    
    def _extract_text(self, element: ET.Element) -> str:
        """Extract text from an XML element, handling CDATA."""
        text = element.text or ""
        return text.strip()
    
    def load_all(self) -> dict[str, PromptTemplate]:
        """Load all available prompt templates."""
        templates = {}
        for name in self.list_prompts():
            try:
                templates[name] = self.load_prompt(name)
            except Exception as e:
                print(f"Error loading prompt '{name}': {e}")
        return templates
    
    def get_prompt_info(self, name: str) -> dict:
        """Get metadata about a prompt template."""
        template = self.load_prompt(name)
        return {
            'name': template.name,
            'version': template.version,
            'file_path': template.file_path,
            'system_prompt_length': len(template.system_prompt),
            'user_template_length': len(template.user_template),
        }
    
    def compare_prompts(self) -> list[dict]:
        """Get comparison info for all prompts."""
        info_list = []
        for name in self.list_prompts():
            try:
                info = self.get_prompt_info(name)
                info_list.append(info)
            except Exception as e:
                info_list.append({'name': name, 'error': str(e)})
        return info_list


if __name__ == "__main__":
    loader = PromptLoader()
    print("Available prompts:")
    for name in loader.list_prompts():
        info = loader.get_prompt_info(name)
        print(f"  - {name} (v{info.get('version', '?')}, {info['system_prompt_length']} chars)")
