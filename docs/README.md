<!--
  Licensed to the Apache Software Foundation (ASF) under one
  or more contributor license agreements.  See the NOTICE file
  distributed with this work for additional information
  regarding copyright ownership.  The ASF licenses this file
  to you under the Apache License, Version 2.0 (the
  "License"); you may not use this file except in compliance
  with the License.  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing,
  software distributed under the License is distributed on an
  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  KIND, either express or implied.  See the License for the
  specific language governing permissions and limitations
  under the License.
-->

# Mosaic Documentation

Static HTML documentation for Paimon Mosaic.

## Local Preview

Serve the docs locally with any static file server:

```bash
# Python
cd docs
python3 -m http.server 8090

# Node.js (npx)
npx serve docs
```

Then open http://localhost:8090 in your browser.

## Structure

```
docs/
├── index.html                        # Home page (overview, benchmarks, types)
├── design.html                       # Format specification & binary layout
├── java-api.html                     # Java API guide
├── python-api.html                   # Python API guide
├── cpp-api.html                      # C++ API guide
├── releases.html                     # Release listing & download links
├── creating-a-release.html           # Release Manager guide
├── verifying-a-release-candidate.html # Verification guide for voters
├── css/
│   └── style.css                     # Styles (light/dark theme)
└── js/
    └── main.js                       # Theme toggle & mobile menu
```

## No Build Step Required

The documentation is plain HTML/CSS/JS with no build toolchain. Edit the HTML files directly and refresh the browser to see changes.

Fonts are loaded from Google Fonts CDN (`DM Sans` and `JetBrains Mono`). An internet connection is needed for fonts to render correctly during local preview.
