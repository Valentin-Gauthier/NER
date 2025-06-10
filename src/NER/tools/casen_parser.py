
class CasENParser:

    def __init__(self, verbose:bool=False):
        self.verbose =  verbose


    def format_tag_recursive(self, content: str) -> str:
        """Traite récursivement le contenu des balises"""
        if '{' in content and '}' in content:
            return self.convert(content)
        return content

    def format_tag(self, raw_tag: str) -> str:
        """Format on tag recursivly"""

        tag = raw_tag[1:-1].replace('\\', '')
        
        if self.verbose:
            print(f"Processing tag: {tag}")
        
        try:
            left, right = tag.rsplit(",.", 1)
        except ValueError:
            return tag  
        
        processed_left = self.format_tag_recursive(left)
        
        parts = right.split("+")
        nametag = parts[0]
        attributes = []
        
        for attr in parts[1:]:
            if '=' in attr:
                key, value = attr.split("=", 1)
                attributes.append(f'{key}={value}')
            elif "grf" in attr:
                attributes.append(f'grf="{attr}"')
            else:
                attributes.append(attr)
        
        attributes_str = " ".join(attributes)
        
        return f'<{nametag} {attributes_str}>{processed_left}</{nametag}>'

    def get_tags(self, text:str) -> list:
        """ Return every tag founds in the text"""
        stack = []
        tags = []
        start_index = None
        
        for i, char in enumerate(text):
            if char == '{':
                if not stack:
                    start_index = i
                stack.append(i)
            elif char == '}' and stack:
                stack.pop()
                if not stack and start_index is not None:
                    tags.append(text[start_index:i+1])
                    start_index = None
        
        return tags

    def convert(self, text:str) -> str:
        """Convert CasEN result text into XML format"""
        tags = self.get_tags(text)
        if not tags:
            return text
        
        for tag in tags:
            new_tag = self.format_tag(tag)
            text = text.replace(tag, new_tag, 1)

        return text