class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class CircularLinkedList:
    def __init__(self):
        self.head = None

    def insert(self, data):
        if not self.head:
            self.head = Node(data)
            self.head.next = self.head
        else:
            new_node = Node(data)
            current = self.head
            while current.next != self.head:
                current = current.next
            current.next = new_node
            new_node.next = self.head

    def get_movies(self):
        movies = []
        if not self.head:
            return movies
        current = self.head
        while True:
            movies.append(current.data)
            current = current.next
            if current == self.head:
                break
        return movies

    def generate_html(self):
        movies = self.get_movies()
        html_code = "<div class='movie-container'>\n"
        for movie in movies:
            html_code += f"    <div class='movie'>\n"
            html_code += f"        <h3>{movie['title']}</h3>\n"
            html_code += f"        <p>{movie['description']}</p>\n"
            html_code += f"        <img src='{movie['imageUrl']}' alt='{movie['title']}'>\n"
            html_code += f"        <a href='{movie['watchUrl']}'>Watch Now</a>\n"
            html_code += f"    </div>\n"
        html_code += "</div>"
        return html_code
